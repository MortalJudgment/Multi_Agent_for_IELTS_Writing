import os
import streamlit as st
from model import CrewAgent
import time
from sugggest_outline import suggest_outline
crew = CrewAgent()

def option1():
    st.title(":blue[Scoring] a Writing IELTS Task 2 📝")
    st.header("Using LLMs to score to your IELTS Writing task 2")
    st.markdown("*It could take around 2 minutes to give the response 😊*")
    with st.container():
        question = st.text_area("Enter the question:", height=80)
        essay = st.text_area("Enter the essay:", height=250)

        if st.button("Score Essay") and question and essay:
            tabs = st.tabs(["Coherence and Cohesion", "Lexical Resource", "Grammatical Range and Accuracy", "Task Response", "Final Response"])
            process = []
            all_response = []
            # Start the scoring process
            start_time_total = time.time()
            output = {"Coherence and Cohesion": '',
                      "Lexical Resource": '',
                      "Grammatical Range and Accuracy": '',
                      "Task Response": ''}
            while True:
                BOSS_command = crew.Manager_Agent(model_name='llama3-8b-8192', previous_task=process)
                process.append(BOSS_command.strip('\'"'))
                start_time = time.time()
                if "Complete" in process:
                    overall_response = crew.Final_output_scoring(question, essay, all_response, model_name='gemini-1.5-pro')
                    break
                else:
                    if "Coherence and Cohesion" in BOSS_command:
                        criteria = "Coherence and Cohesion"
                    elif "Lexical Resource" in BOSS_command:
                        criteria = "Lexical Resource"
                    elif "Grammatical Range and Accuracy" in BOSS_command:
                        criteria = "Grammatical Range and Accuracy"
                    elif "Task Response" in BOSS_command:
                        criteria = "Task Response"
                    else:
                        pass

                    idx = (list(output.keys()).index(criteria))
                    with tabs[idx]:
                        with st.spinner(f"Please wait, it could take some time..."):
                            st.write(f"Scoring {criteria} criteria")
                            score_response, check = crew.Scoring_Agent(question, essay, criteria=BOSS_command, model_name='llama3-70b-8192')
                            if not check == "RESONABLE":
                                score_response, _ = crew.Scoring_Agent(question, essay, criteria=BOSS_command, model_name='llama3-70b-8192', Validate = check)
                            response = crew.Output_Scoring(score_response, question, essay, criteria=BOSS_command, model_name='gemini-1.5-flash-latest')
                            output[criteria] = response
                            st.write(list(output.values())[idx])
                        st.success(f'Complete scoring {criteria} criteria!', icon="✅")
                        all_response.append(response + "\n\n")
                        end_time = time.time() - start_time
                        st.write(f"It's take {end_time:.2f} seconds to scoring {criteria} criteria")
            
            with tabs[4]:
                st.header("Final Response")
                st.write(overall_response)
                end_time_total = time.time() - start_time_total
                st.write(f"It's take {end_time_total:.2f} seconds to give the entire scoring")
            st.button("Rerun")

def option2():
    st.title(":blue[Validate] a Writing IELTS Task 2 📝")
    st.header("Using LLMs to score and give detail feedback to your IELTS Writing task 2")
    st.markdown("*Since it is designed to give detailed feedback, it can take about 4 minutes to give response*.😊")
    with st.container():
        question = st.text_area("Enter the question:", height=80)
        essay = st.text_area("Enter the essay:", height=250)
        if st.button("Score Essay") and question and essay:
            tabs = st.tabs(["Coherence and Cohesion", "Lexical Resource", "Grammatical Range and Accuracy", "Task Response", "Final Response"])
            process = []
            all_response = []
            # Start the scoring process
            output = {"Coherence and Cohesion": '',
                      "Lexical Resource": '',
                      "Grammatical Range and Accuracy": '',
                      "Task Response": ''}
            while True:
                BOSS_command = crew.Manager_Agent(model_name='llama3-8b-8192', previous_task=process)
                process.append(BOSS_command.strip('\'"'))
                start_time = time.time()
                if "Complete" in process:
                    start_time = time.time()
                    with tabs[4]:
                        with st.spinner(f"Please wait, it could take some time..."):
                            overall_response = crew.Output_Agent(question, essay, all_response, model_name='gemini-1.5-pro')
                            st.header("Final Response")
                            st.write(overall_response)
                            st.success(f'Complete!', icon="✅")
                            end_time = time.time() - start_time
                            st.write(f"It's take {end_time:.2f} seconds to give the final response")                        
                    break
                else:
                    if "Coherence and Cohesion" in BOSS_command:
                        criteria = "Coherence and Cohesion"
                    elif "Lexical Resource" in BOSS_command:
                        criteria = "Lexical Resource"
                    elif "Grammatical Range and Accuracy" in BOSS_command:
                        criteria = "Grammatical Range and Accuracy"
                    elif "Task Response" in BOSS_command:
                        criteria = "Task Response"
                    else:
                        pass
                    
                    idx = (list(output.keys()).index(criteria))
                    with tabs[idx]:
                        with st.spinner(f"Please wait, it could take some time..."):
                            st.write(f"Scoring {criteria} criteria")
                            score_response, check = crew.Scoring_Agent(question, essay, criteria=BOSS_command, model_name='llama3-70b-8192')
                            if not check == "RESONABLE":
                                score_response, _ = crew.Scoring_Agent(question, essay, criteria=BOSS_command, model_name='llama3-70b-8192', Validate = check)

                            feedback_response = crew.Feedback_Agent(score_response, question, essay, criteria=BOSS_command, model_name='gemini-1.5-flash-latest')

                            text_part, refined_part = crew.Refine_Agent(score_response, feedback_response, criteria=BOSS_command, question=question, essay=essay, model_name='gemini-1.5-flash')
                            output[criteria] = refined_part
                            st.write(list(output.values())[idx])
                            with st.expander(f"Show Detailed for {criteria} criteria"):
                                st.write(text_part)
                        st.success(f'Complete scoring {criteria} criteria!', icon="✅")
                        end_time = time.time() - start_time
                        st.write(f"It's take {end_time:.2f} seconds to scoring {criteria} criteria")
                    all_response.append(refined_part + "\n\n")
            
            st.button("Rerun")

def option3():
    st.title("An :blue[Instruction] to IELTS Writing Task 2 📑")
    st.markdown("*It could take some time to give response*.😊")
    st.markdown("- Recommend picking gemini(s) models cause they understand Vietnamese better.")
    st.markdown("- Other would give quick response but maybe got trouble with instruction to Vietnamese.")
    st.sidebar.title('Select an LLM')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-pro', 'llama3-70b-8192', 'llama3-8b-8192', 'gemma-7b-it', 'mixtral-8x7b-32768']
    )
    #  (Recommend ⭐)
    st.sidebar.markdown('*Choosing LLM to help you give an instruction to IELTS writing task 2.*')
    question = st.text_area("Enter your IELTS's question:",height=80)
    # Create a place for users to input a number using a slider
    selected_number = st.slider("Select desired score:", min_value = 6.0, max_value = 9.0, step = 0.5)

    # Display the selected number
    if st.button("Submit"):
        with st.spinner("Please wait ..."):
            response = suggest_outline(question, model, selected_number)
            st.markdown(response)
        st.success("Believe you can and you're halfway there!")

def introduction():
    """
    An introduction guild to user
    """
    st.title("Welcome to IELTS Writing Task 2 Assistant")
    st.header("This app is designed to help you with your IELTS Writing Task 2 preparation 📝")
    st.markdown("Here's what you can do:")
    st.markdown("👉 **Score your essays:** Get a detailed score breakdown for your essays based on the four IELTS criteria.")
    st.markdown("👉 **Validate your essays:** Get detailed feedback on your essays, including suggestions for improvement.")
    st.markdown("👉 **Get an instruction:** Get an instruction for your essays based on the question and your desired score.")
    st.markdown("Hoping this help you with your upcoming IELTS test.")
    st.markdown("Let's get started! 🤘")


def main():
    st.set_page_config(page_title='IELTS Wrinting task 2', layout='wide', page_icon='📝', initial_sidebar_state='auto')

    st.sidebar.title('Select a Task')
    option = st.sidebar.selectbox("Select an option:", ["Introduction", "Scoring IELTS Writing task 2", "Validate IELTS Writing task 2", "Give an instruction to IELTS Writing task 2"])

    if option == "Introduction":
        introduction()
    elif option == "Scoring IELTS Writing task 2":
        option1()
    elif option == "Validate IELTS Writing task 2":
        option2()
    elif option == "Give an instruction to IELTS Writing task 2":
        option3()       

if __name__ == "__main__":
    main()
