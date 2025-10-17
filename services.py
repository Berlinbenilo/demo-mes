import os
from typing import Dict, Optional

import face_recognition
from google import genai
from google.genai import types
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from pydantic import Field

from properties import endpoint, azure_key, entities_to_find, entity_mapping

gemini_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
azure_mini = AzureChatOpenAI(**{
    'temperature': 0,
    "model": "gpt-4o-mini",
    "azure_endpoint": endpoint,
    "api_key": azure_key,
    "api_version": "2025-01-01-preview",
})


class LLMWrapper:
    def __init__(self, llm=None):
        self.llm = llm

    def invoke_with_parser(self, prompt_template: str, llm=None, placeholder_input: Dict = None,
                           validator: Optional[BaseModel] = None, stream: bool = False):
        _parser = JsonOutputParser(pydantic_object=validator) if validator else StrOutputParser()
        if not prompt_template:
            prompt_template = "You are a helpful assistant. Answer the following question:\n{query}"
        prompt = PromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": _parser.get_format_instructions() if validator else ""}
        )

        llm_instance = llm or self.llm
        chain = prompt | llm_instance | _parser

        if stream:
            return self._invoke_streaming(prompt, llm_instance, placeholder_input or {})

        try:
            return chain.invoke(placeholder_input or {})
        except Exception as e:
            raise

    @staticmethod
    async def _invoke_streaming(prompt, llm, input_data: Dict):
        prompt_value = prompt.format(**input_data)
        async for chunk in llm.astream(prompt_value):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield token

client = genai.Client()
llm_wrapper = LLMWrapper(llm = azure_mini)

class ParserOutput(BaseModel):
    entity: Optional[str] = Field(default=None, description="Matched entity from the provided list")



def transcript_audio(audio_file_path: str, prompt: str) -> Dict:
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            'Please transcribe this audio file verbatim',
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type='audio/mp3',
            )
        ]
    )
    prompt_template = """You are the entity recogniser your task is to find the appropriate entity from the given 
    transcription text if iot matches the list of entity. The text is {text}. The list of entities are {entities}. 
    Return the output in the json format as per given below, {format_instructions}. Note" If nothing is matched leave it empty """""
    output = llm_wrapper.invoke_with_parser(prompt_template= prompt_template,
                                         placeholder_input={"text": response.text,
                                                            "entities": entities_to_find},
                                         validator = ParserOutput)
    return {"transcription": response.text, "target_url":entity_mapping.get(output.get('entity', None), None)}


def setup_face_recognition(known_face_dir: str, upload_dir: str):
    if not os.path.exists(known_face_dir):
        os.makedirs(known_face_dir)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Load all known faces
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_face_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(known_face_dir, filename)
            name = os.path.splitext(filename)[0]  # filename without extension
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            else:
                print(f"⚠️ No face found in {filename}, skipping...")
    return known_face_encodings, known_face_names
