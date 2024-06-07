import sys
import os
import re
import time
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from groq import Groq
from prompt import Band_description, ScoringPrompt, RefinedOutput, Output_Scoring
from langchain.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.tools import StructuredTool

import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()
# Groq
try:
    # Load API keys from environment variable
    groq_api_key = os.environ["GROQ_API_KEY"]
    groq_llm = Groq(api_key=groq_api_key)
except Exception as exception:
    print("Error in initializing API keys:", exception)
    raise  
# Google gemini
try:
    # Load API keys from environment variable
    google_api_key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)
except Exception as exception:
    print("Error in initializing API keys:", exception)
    raise

console = Console()
validate_prompt = Band_description()
scoring_prompt = ScoringPrompt()
output_refined = RefinedOutput()
output_scoring = Output_Scoring()

def generate_response(prompt:str, model, temperature = 0.3, max_output = 32768):

  # Set up the model
  generation_config = {
    "temperature": temperature,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": max_output,
  }
  
  model = genai.GenerativeModel(model_name = model,generation_config=generation_config)
    
  try:
    response = model.generate_content(prompt)
    return response.text
  except Exception as exception:
    print("Error generating response:", exception)

## Validate tools
validate_template = """This is score given from 'score_agent'
{score}

Your job is compare that score with score description given:
{band_description}

Be wise, honest and clever.
If score given from 'score_agent' reasoning to score description then return 'RESONABLE'.
If score given from 'score_agent' not reasoning to score description, then return information about the sub-criterias are not reasoning so that scoring agent can score again to get a better result and suggest score agent to HIGHER or LOWER score. The sub-criterias are already reasoning should not mention.
The returned information must be valuable, focusing on the criteria that were evaluated inappropriately.
Please wise, smart and do not response unecessary information.
"""
prompt = PromptTemplate(input_variables=["score", "band_description"], template=validate_template)
def LLM_chain(score, band_description, prompt):
    model = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"score": score, "band_description": band_description})

class ValidateInput(BaseModel):
    score: str = Field(description="score")
    band_description: str = Field(description="chat history")

def validate(score: str, band_description: str) -> str:
    """Validate the score"""
    return LLM_chain(score, band_description, prompt)


validate_tool = StructuredTool.from_function(
    func=validate,
    name="Validate the score",
    description="Summary the chat",
    args_schema=ValidateInput,
    return_direct=True,
)


class CrewAgent():
    def Manager_Agent(self, model_name, previous_task):
        if previous_task == []:
            previous_task = "Let's start!"
            console.print("\n[bold yellow]Let's start Evaluating IELTS Writing task 2[/bold yellow]")

        system_prompt = """You are the manager of a group of AI Agents tasked with evaluating IELTS Writing Task 2.
                        The team needs to evaluate based on 4 criteria: 
                        - "Coherence and Cohesion"
                        - "Lexical Resource"
                        - "Grammatical Range and Accuracy"
                        - "Task Response"

                        Your job is to let the team know which criteria they should evaluate next to complete the task.
                        If the task has just begun, return the first criteria: "Coherence and Cohesion".
                        If all criteria have been evaluated as indicated in the user prompt, you MUST return "Complete".
                        You MUST NOT return a criterion that has already been done.
                        Your return should be a single criterion or "Complete" at a time (do not give other words or information).
                        """ 
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are the criterias have been evaluated:\n {previous_task}.\n\n If all the criteria is done then return 'Complete'. Otherwise, return next criteria"},
        ]

        response = groq_llm.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1000,
        )

        response_text = response.choices[0].message.content
        if "Complete" in response_text:
            console.print(f"\n[bold yellow]Complete validate the essay. Summary to get final result[/bold yellow]")
        else:
            console.print(f"\n[bold yellow]Scoring {response_text} criteria[/bold yellow]")
        return response_text


    def Scoring_Agent(self, question, essay, criteria, model_name, Validate = ''):
        console.print("\n[bold green]Calling Score Agent[/bold green]")

        if "Coherence and Cohesion" in criteria:
            description, sub_criterias = scoring_prompt.Coherence_and_Cohesion()
            band_description = validate_prompt.Coherence_and_Cohesion()
        elif "Lexical Resource" in criteria:
            description, sub_criterias = scoring_prompt.Lexical_Resource()
            band_description = validate_prompt.Lexical_Resource()
        elif "Grammatical Range and Accuracy" in criteria:
            description, sub_criterias = scoring_prompt.Grammatical_Range_and_Accuracy()
            band_description = validate_prompt.Grammatical_Range_and_Accuracy()
        elif "Task Response" in criteria:
            description, sub_criterias = scoring_prompt.Task_Response()
            band_description = validate_prompt.Task_Response()
        else:
            sys.exit("ERROR on Criteria name")

        system_message = f"""You are an expert AI agent that scoring the {criteria} criteria of an IELTS Writing Task 2 essay. 
        To get the {criteria} criteria score, you must score its sub-criterias. (NOTE THAT!!! The sub-criterias score MUST be an integer in range 0-9)
        {description}\n
        Think step by step, for each sub-criterias do the following flow: list strong points, weak points then give score.
        Then the overall {criteria} score MUST equal to the lowest sub-criteria score. (NOTE THAT!!! The criteria score MUST be an integer in range 0-9)"""
        
        context = f"""Given an IELTS writing Task 2 \n
        Question: '{question}'
        Essay: '{essay}'
        Sub-criterias are: 
        {sub_criterias}
        """

        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": context
            }
        ]
        if Validate != '':
            messages[1]["content"] += f"This is analysis of your scoring last time compared to the band description.\n\n{Validate}.\n You MUST use this information to score again (based on advice score) to get the correct result. Since you score again, you still give strong points and weak points also."

        response = groq_llm.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=3000,
        )

        response_text = response.choices[0].message.content
        check = validate_tool.run({"score": response_text, "band_description": band_description})

        return response_text, check

    def Feedback_Agent(self, score_response, question, essay, criteria, model_name):
        console.print("\n[bold purple]Generating feedback[/bold purple]")

        message = f"""You are an AI assistant give feedback for an IELTS Writing Task 2 essay. 
        You will be given an information for a {criteria} criteria score.
        Your job is using those information to give improverment suggestion (about 300 words) for each sub-criterias and one general feedback (about 120 words) to the Vietnamese user.
        I need the response in a natural and friendly tone for a Vietnamese user. KEEP correctness, honest and wise.
        NOTE that: 
        - You have around 2000 words total so give valuable and necessary advices.
        - Keep the criteria, sub-criterias and words quoted in the passage in English, the rest should be in Vietnamese.
        - The improverment suggestion and feedback must based on the orgin essay.
        - For the improverment suggestion, focus on weakpoints, if the essay is good enough then improverment suggestion can be ignored.

        REMEBER: !!! DO NOT LIST all Linking Words. If you try to give improverment or feedback on Linking Words, just give a few Linking Words is enough.
        Origin question '{question}' 
        Origin essay: '{essay}'
        The given Score {score_response}
        """
        
        
        response_text = generate_response(message, model = model_name, temperature = 0.3, max_output = 8192)
        return response_text

    def Refine_Agent(self, score_response, feedback, criteria, question, essay, model_name):
        console.print("\n[bold blue]Refining output[/bold blue]")

        if "Coherence and Cohesion" in criteria:
            criteria = 'Coherence and Cohesion'
            prompt = output_refined.Coherence_and_Cohesion()
        elif "Lexical Resource" in criteria:
            criteria = 'Lexical Resource'
            prompt = output_refined.Lexical_Resource()
        elif "Grammatical Range and Accuracy" in criteria:
            criteria = 'Grammatical Range and Accuracy'
            prompt = output_refined.Grammatical_Range_and_Accuracy()
        elif "Task Response" in criteria:
            criteria = 'Task Response'
            prompt = output_refined.Task_Response()
        else:
            sys.exit("ERROR on Criteria name")

        message = f"""You are an AI assistant compiling a comprehensive report for an IELTS Writing Task 2 essay. 
        You will be given an information for a {criteria} criteria evaluation.
        Your job is using those information to give final response to the Vietnamese user.
        I need the response in a natural and friendly tone for a Vietnamese user. KEEP correctness, honest and wise.
        NOTE that: 
        - Your response should design for Vietnamese user, strong points and weak points should translate to Vietnamese also.
        - Keep the criteria, sub-criterias and words quoted in the essay in English, the rest should be in Vietnamese.
        - Score for criteria and sub-criterias MUST not change.
        - The improverment suggestion and feedback must close to the essay.
        - The correction only in esscessary and when the {criteria} criteria in low score.
        - The correction must relate to {criteria} criteria or its sub-criteria
        - Give a Refined part which is the part includes shorten information of the task.
        - The Refined part include: {criteria} criteria score, general improverment suggestion (Atleast 100 words Prioritize low score sub-criteria) and feedback.
        Origin question '{question}' 
        Origin essay: '{essay}'

        The given Score {score_response}
        The given improverment suggestion and feedback:
        {feedback}
        Expected output:
        {prompt}
        ## REFINED PART:
        **{criteria}** criteria score: 
        - General improverment: 
            -
            -

        - Feedback:
            -
        """
        
        response = generate_response(message, model = model_name, temperature = 0.3, max_output = 65536)
            
        parts = re.split(r'## REFINED PART:', response)

            # Assigning the split parts to text and refined variables
        text_part = parts[0].strip()
        refined_part = parts[1].strip()            
        
        return text_part, refined_part
    
    def Output_Agent(self, question, essay, context, model_name):
        console.print("\n[bold blue]Giving final output[/bold blue]")
        message = f"""You are an AI assistant compiling a comprehensive report for an IELTS Writing Task 2 essay. 
        Origin question: {question}
        Origin essay: {essay}
        You will be given an information for all criteria evaluation. 
        Your job is compute Overall band score of IELTS Writing Task 2 for given 4 criterias score (given below).
        Your job is give correction to the essay by following rule:
            - You return all the origin esssay.
            - The structure of the origin esssay may be wrong due to user input. You should refactor appropriately to IELTS writing task 2 essay
            - You MUST NOT change any words in the origin esssay.
            - If words incorrect(in grammar term), you should give it into :red[], and the correction word in blanket next to it.
            - If words are repeated, you should give it into :orange[], and the synonyms can be substituted to avoid repetition in blanket next to it.
            - If words you find can replace to get better result, you should give it into :violet[], and the replace words in blanket next to it.
            - If words you find good, you should give it into :green[].
        For example,
        '''
        In **:orange[several]**(many) nations, the majority of **:orange[influencers]**(famous people) are having an effect on young generations. The writer contends that this phenomenon brings **:violet[drawbacks]**(negative effects) to the youth due to the imitation of children as well as conveying negative ideologies of famous people.

        It is crucial to understand that children always follow the **:orange[attitude]**(behavior) of adults. To be more specific, the youth watch many videos on the Internet and cannot identify between the negative and positive behaviors of well-known individuals. Indeed, youngsters are unable to control their **:red[impulsion]**(impulses), therefore, they just imitate the influencer's behaviors. In some cases, many bad attitudes are popularized on social platforms, then children believe that is the trend to catch up with this. Consequently, children become **:red[unwell]**(problematic) in society or even commit an offense, leading to a detrimental impact on social safety.

        **:green[Furthermore]**, some well-known people have popularized their wrong ideology about a specific field of life, especially **:red[policy]**(politics). To explain in more detail, these kinds of information sink into children's thoughts and they cannot be aware of its danger. To a certain extent, the youth follow this knowledge and support the extremist parties. For this reason, children argue more about political problems in their nation and **:red[against]**(oppose) the government. Thus, these young generations become marginalized in society.

        **:green[In conclusion]**, exposure to well-known people in the younger generations is a negative development because of an ability to mimic negative behavior and wrong ideas in political problems. Hence, it has been shown that parents should avoid children watching well-known people.

        '''
        Moreover, your job is give improverment to get a better result for user's essay.
        For example,
        '''
        - "gain insight from their failures" - Mặc dù cụm từ này không nhất thiết là sai nhưng nó có thể nghe khó hiểu hoặc không tự nhiên đối với một số độc giả. Một cách phổ biến hơn để thể hiện ý tưởng này có thể là "learn from their mistakes."
        - Trong đoạn văn thứ hai, câu "When children are given the freedom to choose the activities in which they participate, they are more likely to do so with excitement and motivation, which ultimately leads to a feeling of success and fulfillment" mặc dù không hề sai về mặt cấu trúc ngữ pháp, sử dụng nhiều mệnh đề phụ, khiến cho việc đọc hiểu có phần khó khăn. Có thể được chia thành hai câu cho rõ ràng. Các câu sửa lại phải là: "When children are given the freedom to choose the activities in which they participate, they are more likely to do so with excitement and motivation. This ultimately leads to a feeling of success and fulfillment."
        '''
        REMEMBER: 
        - Words quoted from the origin essay MUST keep in English
        - If more than one improverment, you should break it down to make it easier to read.
        - The improverment should not too far from user overall bandscore. We encourage users to improve the quality of their essay gradually.
        NOTE that The response is designed to the Vietnamese user. Present in order to easy to read and print out in markdown.
        Given context: {context}

        Expected ouput:
        1. Coherence and Cohesion criteria: score
        - General improverment:
            -
        - Feedback:
            -

        2. Lexical Resource criteria: score                        
        - General improverment:
            -
        - Feedback:
            -
        
        3. Grammatical Range and Accuracy criteria: score                          
        - General improverment:
            -
        - Feedback:
            -

        4. Task Response criteria: score                            
        - General improverment:
            -
        - Feedback:
            -

        **Overall band score**: score

        ##### Correction (if need):
        -
        - Label note
            - **:red[red]** for incorrect word.
            - **:orange[orange]** for repeat word.
            - **:violet[violet]** for should replace word.
            - **:green[green]** for good word.

        ##### Improverment (if have):
        -
        """

        response = generate_response(message, model = model_name, temperature = 0.3, max_output = 65536)

        final_report = response
       
        return final_report
    
    def Output_Scoring(self, score_response, question, essay, criteria, model_name):
        if "Coherence and Cohesion" in criteria:
            prompt = output_scoring.Coherence_and_Cohesion()
        elif "Lexical Resource" in criteria:
            prompt = output_scoring.Lexical_Resource()
        elif "Grammatical Range and Accuracy" in criteria:
            prompt = output_scoring.Grammatical_Range_and_Accuracy()
        elif "Task Response" in criteria:
            prompt = output_scoring.Task_Response()
        else:
            sys.exit("ERROR on Criteria name")

        message = f"""You are an AI assistant compiling a comprehensive report for an IELTS Writing Task 2 essay. 
        You will be given an information for a {criteria} criteria evaluation.
        Your job is using those information to give final response to the Vietnamese user.
        I need the response in a natural and friendly tone for a Vietnamese user. KEEP correctness, honest and wise.
        NOTE that: 
        - Your response should design for Vietnamese user, strong points and weak points should translate to Vietnamese also.
        - Keep the criteria, sub-criterias and words quoted in the essay in English, the rest should be in Vietnamese.
        - Score for criteria and sub-criterias MUST not change. 
        Origin question '{question}' 
        Origin essay: '{essay}'

        The given Score {score_response}
        Expected output:
        {prompt}
        """
        
        response = generate_response(message, model = model_name, temperature = 0.3, max_output = 65536)

        final_report = response
        console.print(
            Panel(
                Markdown(final_report),
                title="[bold blue]Refined[/bold blue]",
                title_align="left",
                border_style="blue",
            )
        )
        return final_report
    
    def Final_output_scoring(self, question, essay, context, model_name):
        console.print("\n[bold blue]Giving final output[/bold blue]")
        message = f"""You are an AI assistant compiling a comprehensive report for an IELTS Writing Task 2 essay. 
        Origin question: {question}
        Origin essay: {essay}
        You will be given an information for all criteria scoring.
        Your job is compute Overall band score of IELTS Writing Task 2 for given 4 criterias score (given below).
        Your job is give short improverment tip or feedback for user to improve their score (should close to the question and essay).
        REMEMBER: Words quoted from the origin essay MUST keep in English
        You should present friendly for user to easy to read.
        The improverment should not too far from user overall bandscore. We encourage users to improve the quality of their essay gradually.
        NOTE that The response is designed to the Vietnamese user. KEEP natural and friendly tone.
        Given context: {context}
        Expected ouput:
        ### Criteria score:
        **Coherence and Cohesion** criteria: (score)
        **Lexical Resource** criteria: (score)                            
        **Grammatical Range and Accuracy** criteria: (score)                             
        **Task Response** criteria: (score)                            
        ### **Overall band score**: (score)
        #### Improverment tip (if have):
        -
        """

        response = generate_response(message, model = model_name, temperature = 0.3, max_output = 65536)

        final_report = response
        console.print(
            Panel(
                Markdown(final_report),
                title="[bold blue]Final Report[/bold blue]",
                title_align="left",
                border_style="blue",
            )
        )
        return final_report
    

if __name__ == "__main__":
    
    A_MODEL = "llama3-70b-8192"     # 30 request/min
                                    # 6,000 tokens/min
                                    # Context Window: 8,192 tokens

    B_MODEL = "llama3-8b-8192"      # 30 request/min
                                    # 30,000 tokens/min
                                    # Context Window: 8,192 tokens

    C_MODEL = "mixtral-8x7b-32768"  # 30 request/min
                                    # 5,000 tokens/min
                                    # Context Window: 32,768 tokens

    D_MODEL = "gemini-pro"    

    E_MODEL = "gemini-1.5-flash"    # 15 request/min
                                    # 1 million tokens/min
                                    # Context Window: 1,048,576 tokens

    F_MODEL = "gemini-1.5-flash-latest"     # 15 request/min
                                            # 1 million tokens/min
                                            # Context Window: 1,048,576 tokens

    G_MODEL = "gemini-1.5-pro"              # 2 request/min
                                            # 32,000 tokens/min
                                            # Context Window: 1,048,576 tokens


    question = "The increase in the production of consumer goods results in damage to the natural environment. What are the causes of this? What can be done to solve this problem?"
    essay = """Nowadays, as more consumer goods are manufactured, more damage has been inflicted on the environment. I will outline several reasons for this and put forward some measures to this issue.

    First of all, the increase in the production of consumer products harms the environment in two ways: the chemical by-products from the manufacturing process and the mass production of disposable goods. As more goods are produced, more toxic wastes and emissions are released from factories into nature. Water sources are contaminated, and the air is severely polluted, which results in the deaths of many marine and terrestrial animals. Also, to accommodate customers’ ever-increasing demands, more single-use products are introduced, most of which are non-biodegradable. Though having a short lifespan, these products can remain as wastes for thousands of years, turning our planet into a huge landfill and posing a threat to the living habitats of all creatures.

    Actions must be taken as soon as possible to minimize the negative impacts on the environment arising from the increasing amount of consumer goods. First, companies should promote the use of eco-friendlier materials. For example, the giant coffee chain Starbucks has recently replaced plastic straws with reusable alternatives made of materials like paper or bamboo. In addition, many governments are also encouraging the development of more sustainable manufacturing processes. For instance, many states in the U.S offer tax breaks and incentives for businesses using renewable energy, and some firms are even allowed to purchase green energy at cheaper prices than traditional fossil fuels.

    In conclusion, there are two main reasons why the environment is severely impacted by the increase in the production of consumer goods. To address this issue, governments and companies must join hands to make the production lines more environmentally friendly by switching to greener materials."""

    crew = CrewAgent()

    process = []
    all_response = []
    start_time = time.time()
    while True:
        BOSS_command = crew.Manager_Agent(model_name = B_MODEL, previous_task = process)
        process.append(BOSS_command.strip('\'"'))
        if "Complete" in process:
            overall_response = crew.Output_Agent(question, essay, all_response, model_name = G_MODEL)
            console.print(  
                Panel(
                    Markdown(overall_response),
                    title="[bold blue]Final Report[/bold blue]",
                    title_align="left",
                    border_style="blue",
                )
            )
            break
        else:
            score_response, check = crew.Scoring_Agent(question, essay, criteria = BOSS_command , model_name=A_MODEL, Validate = '')
            console.print(
                Panel(
                    Markdown(score_response),
                    title=f"[bold green]Scoring {BOSS_command} [/bold green]",
                    title_align="left",
                    border_style="green",
                    subtitle="Completed detailed scoring, sending result to next step 👇",
                )
            )
            if "RESONABLE" not in check:
                console.print(
                    Panel(
                        "Since score not RESONABLE. Please wait to re-score",
                        title="[bold red]Final Report[/bold red]",
                        title_align="left",
                        border_style="red",
                    )
                )
                score_response, _ = crew.Scoring_Agent(question, essay, criteria = BOSS_command , model_name=A_MODEL, Validate = check)
                console.print(
                    Panel(
                        Markdown(score_response),
                        title=f"[bold green]Scoring {BOSS_command} [/bold green]",
                        title_align="left",
                        border_style="green",
                        subtitle="Completed detailed scoring, sending result to next step 👇",
                    )
                )
            feedback_response = crew.Feedback_Agent(score_response, question, essay, criteria  = BOSS_command , model_name=E_MODEL)
            console.print(
                Panel(
                    Markdown(feedback_response),
                    title="[bold purple]Feedback[/bold purple]",
                    title_align="left",
                    border_style="purple",
                    subtitle="Feedback generated, sending to final refinement 👇",
                )
            )
            text_part, refined_part = crew.Refine_Agent(score_response, feedback_response, criteria = BOSS_command, question = question, essay = essay, model_name=F_MODEL)
            console.print(
                Panel(
                    Markdown(text_part),
                    title="[bold blue]Refined[/bold blue]",
                    title_align="left",
                    border_style="blue",
                )
            )
            all_response.append(refined_part + "\n\n")
    end_time = time.time() - start_time
    print(end_time)
