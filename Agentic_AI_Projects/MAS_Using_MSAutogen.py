import os
from dotenv import load_dotenv
import pprint
env_path = r'C:\AI\.env'
load_dotenv(env_path)

# Configure
config_list = [{
    "model": "llama-3.3-70b-versatile",
    "api_key": os.environ.get("GROQ_API_KEY"),
    "api_type": "groq"
}]

# Warning control
import warnings
warnings.filterwarnings('ignore')

from autogen import ConversableAgent, AssistantAgent, UserProxyAgent
from autogen import initiate_chats

agent = ConversableAgent(
    name="chatbot",
    llm_config={"config_list": config_list},
    human_input_mode="NEVER",
)

reply = agent.generate_reply(
    messages=[{"content": "Tell me a joke.", "role": "user"}]
)
print(reply['content'])

# ConversableAgent - ConversableAgents are designed for structured conversations.(example - onboarding)
# AssistantAgent - AssistantAgent are task oriented agents, often having expertise 
# UserProxyAgent - act as intermediaries between the user and the system. They relay user inputs

# Define the RegistrationAgent for collecting user details
registration_agent = ConversableAgent(
    name="User Registration Agent",
    system_message='''You are a friendly registration assistant.
    Your task is to collect the user's full name, company name, and the product they are interested in.
    Do not ask for additional details. Respond with 'END_SESSION' once all details are obtained.''',
    llm_config={"config_list": config_list},
    code_execution_config=False,
    human_input_mode="NEVER",
)

# Define the HelpDeskAgent for troubleshooting
help_desk_agent = AssistantAgent(
    name="Help Desk Agent",
    system_message='''You are a knowledgeable support assistant.
    Your role is to assist users in buying our product.
    Offer clear steps, share useful resources, and escalate if required.
    Respond with 'END_SESSION' once the issue is addressed.''',
    llm_config={"config_list": config_list},
    code_execution_config=False,
    human_input_mode="NEVER",
)

# Define the SalesAgent for product recommendations
sales_agent = AssistantAgent(
    name="Sales Recommendation Agent",
    system_message='''You are a sales assistant.
    Based on user information, suggest the most suitable product and provide additional details.
    If the user expresses interest, proceed with gathering purchase intent details.
    Respond with 'END_SESSION' when user decides to proceed or not.''',
    llm_config={"config_list": config_list},
    code_execution_config=False,
    human_input_mode="NEVER",
)

# Define the RelayAgent for communication handling
relay_agent = UserProxyAgent(
    name="Interaction Relay Agent",
    system_message='''You act as a bridge between users and specialized agents.
    You do not process requests yourself but ensure smooth communication.''',
    llm_config=False,
    code_execution_config=False,
    human_input_mode="ALWAYS",
)

conversation_flow = [
    {
        "sender": registration_agent,
        "recipient": relay_agent,
        "message": "Hi! I'm here to help with your onboarding. Please provide your full name, company, and the product you're interested in.",
        "summary_method": "reflection_with_llm",
        "summary_args": {
            "summary_prompt": "Extract user details as JSON: {'full_name': '', 'company': '', 'product': ''}",
        },
        "max_turns": 2,
        "clear_history": True
    },
    {
        "sender": relay_agent,
        "recipient": sales_agent,
        "message": "Based on user details, suggest a suitable product.",
        "summary_method": "reflection_with_llm",
        "summary_args": {
            "summary_prompt": "Summarize recommended products as JSON: {'recommended_product': '', 'justification': ''}",
        },
        "max_turns": 3,
        "clear_history": False
    },
    {
        "sender": relay_agent,
        "recipient": help_desk_agent,
        "message": "Based on sale recommendation, closen the conversation with formal sale pitch",
        "summary_method": "reflection_with_llm",
        "summary_args": {
            "summary_prompt": "Summarize the assistance provided as JSON: {'resolution_steps': []}",
        },
        "max_turns": 3,
        "clear_history": False
    }
]

# Execute the conversations
session_results = initiate_chats(conversation_flow)
pprint.pp(session_results)