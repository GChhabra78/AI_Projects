# Load the .env file from local
import openai
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool

env_path = r'C:\AI\.env'
load_dotenv(env_path)
from crewai import Agent, Task, Crew

#Define first Agent(Level 1)
customer_assist_agent = Agent(
    role="Customer Care Specialist",
    goal="Deliver the most courteous and effective "
         "customer support on your team.",
    backstory=(
        "You are a valued team member at Crew AI (https://docs.crewai.com/introduction) "
        "and currently assigned to assist {customer}, a crucial client "
        "for the company."
        "Your mission is to ensure the highest quality of support!"
        "Provide detailed and thorough responses, "
        "avoiding any assumptions."
    ),
    allow_delegation=False,
    verbose=True
)

#Define second agent(Level 2)
quality_check_agent = Agent(
    role="Support Quality Analyst",
    goal="Be recognized for ensuring the highest "
         "standards of support quality in your team.",
    backstory=(
        "You are a key contributor at CrewAI (https://docs.crewai.com/introduction), "
        "working alongside your team to review a support request from {customer}. "
        "Your responsibility is to ensure that the Customer Care Specialist "
        "delivers top-notch service.\n"
        "It’s your duty to confirm that all responses are thorough, "
        "accurate, and free of assumptions."
    ),
    verbose=True
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/introduction",config={}
)

customer_query_task = Task(
    description=(
        "{customer} has submitted a critical request:\n"
        "{inquiry}\n\n"
        "{person} from {customer} initiated the inquiry. "
        "Leverage all available resources and knowledge to "
        "deliver outstanding support."
        "Your goal is to provide a thorough and accurate resolution "
        "to the customer's question."
    ),
    expected_output=(
        "A comprehensive and well-researched response to the customer's inquiry "
        "that addresses every aspect of their concern.\n"
        "The response must reference all the tools and information sources used, "
        "including external materials or solutions. "
        "Ensure that the answer is exhaustive, leaves no room for follow-up questions, "
        "and maintains a professional, approachable tone."
    ),
    tools=[docs_scrape_tool],
    agent=customer_assist_agent,
)

response_review_task = Task(
    description=(
        "Evaluate the response prepared by the Customer Care Specialist for {customer}'s inquiry. "
        "Ensure the reply meets the highest standards of accuracy, completeness, and clarity "
        "expected in customer service.\n"
        "Confirm that every aspect of the customer's question has been addressed "
        "in a friendly and approachable manner.\n"
        "Verify the inclusion of appropriate references and sources used "
        "to gather the information, ensuring the response is well-supported and "
        "leaves no loose ends."
    ),
    expected_output=(
        "A polished, detailed, and ready-to-send response that completely addresses "
        "the customer's inquiry.\n"
        "The response should reflect all necessary feedback and adjustments while "
        "maintaining a relaxed yet professional tone that aligns with our cool and casual company culture."
    ),
    agent=quality_check_agent,
)

crewCSMAS = Crew(
  agents=[customer_assist_agent, quality_check_agent],
  tasks=[customer_query_task, response_review_task],
  verbose=True,
  memory=True
)

inputs = {
    "customer": "Unfold data science",
    "person": "Aman Kumar",
    "inquiry": "I need help with Crew AI "
               "and Understanding about Crew and Agent "
               "How to use Crew and Agent in CrewAI "
               "Can you provide guidance?"
}
result = crewCSMAS.kickoff(inputs=inputs)