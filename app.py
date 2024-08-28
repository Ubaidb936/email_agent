from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

email_prompt = """
  You are an email writer. Use the following input to draft an email:
  Input: {input}
  Deliver:
  1. A complete email.
"""

class Email(BaseModel):
  email: str = Field(description= "email")
email_parser = JsonOutputParser(pydantic_object=Email)

@chain
def email_model(inputs: dict) -> str | list[str] | dict:
 model = ChatOpenAI(temperature=0.5, model="gpt-4o", max_tokens=1024)
 msg = model.invoke(
             [HumanMessage(
             content=[
             {"type": "text", "text": inputs["prompt"]},
             {"type": "text", "text": inputs["parser"].get_format_instructions()},
             ])]
             )
 return msg.content


def get_email(user_input) -> dict:
   parser = email_parser
   prompt = email_prompt.format(input=user_input)
   intent_chain = email_model | parser
   return intent_chain.invoke({'prompt': prompt, 'parser':parser})


import gradio as gr

def process_text(input_text):
    output =  get_email(input_text)
    return output["email"]

# Create the Gradio interface
interface = gr.Interface(
    fn=process_text,           # Function to process the text
    inputs=gr.Textbox(label = "Email Instructions"),       # Textbox input for the user
    outputs=gr.Textbox(label = "Email"),      # Textbox output for the response
    title="Email Writer",    # Title of the app
    # description="Enter email instructions"
)

# Launch the app
interface.launch()
