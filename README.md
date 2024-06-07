## IELTS Writing Task 2 Assistant

This repository contains the code for an IELTS Writing Task 2 assistant powered by LLMs. This assistant can:

- **Score your essays:** Get a detailed score breakdown for your essays based on the four IELTS criteria (Coherence & Cohesion, Lexical Resource, Grammatical Range & Accuracy, Task Response). The assistant provides a score for each sub-criteria and an overall score for the criteria.
- **Validate your essays:** Get detailed feedback on your essays, including suggestions for improvement. The assistant analyzes your essay and identifies areas where you can improve your score. It provides specific feedback on each criteria and suggests ways to improve your writing.
- **Get an instruction:** Get an instruction for your essays based on the question and your desired score. The assistant helps you plan your essay by providing a detailed outline, including:
    - Explaining the task requirements
    - Identifying the essay type (Opinion, Discussion, Problem-Solution, Advantage-Disadvantage)
    - Outlining the essay structure (Introduction, Body Paragraphs, Conclusion)
    - Providing planning tips
    - Highlighting key skills (Clarity, Relevance, Logical Sequencing, Vocabulary, Grammar, Cohesion)
    - Encouraging practice and feedback

### Requirements

- Python 3.8 or higher
- `pip install -r requirements.txt`

### Usage

1. **Set up API keys:**
   - Create a `.env` file in the root directory of the project.
   - Add your API keys for Groq and Google Gemini:
     ```
     GROQ_API_KEY=your_groq_api_key
     GOOGLE_API_KEY=your_google_api_key
     ```
   - You can obtain API keys from the respective platforms.
2. **Run the app:**
   - `streamlit run app.py`

### Project Structure

- **`model.py`:** Contains the core logic for scoring, validating, and refining essays. This file defines a `CrewAgent` class that handles the different tasks.
    - **`Manager_Agent`:** Determines which criteria to evaluate next based on the progress made.
    - **`Scoring_Agent`:** Scores the essay based on a specific criteria using a Groq LLM.
    - **`Feedback_Agent`:** Generates detailed feedback and improvement suggestions for a specific criteria using a Google Gemini LLM.
    - **`Refine_Agent`:** Refines the output of the `Feedback_Agent` to provide a more concise and user-friendly response.
    - **`Output_Agent`:** Generates the final report, including the overall score, corrections, and improvement suggestions, using a Google Gemini LLM.
- **`app.py`:** Streamlit app that provides the user interface. This file defines three main options:
    - **Scoring IELTS Writing task 2:** Allows users to score their essays and get detailed feedback.
    - **Validate IELTS Writing task 2:** Provides detailed feedback and improvement suggestions for essays.
    - **Give an instruction to IELTS Writing task 2:** Generates an instruction based on the question and desired score.
- **`prompt.py`:** Defines prompts for different tasks. This file contains classes for:
    - **`Band_description`:** Defines the band descriptions for each criteria.
    - **`ScoringPrompt`:** Defines the prompts for the `Scoring_Agent`.
    - **`RefinedOutput`:** Defines the expected output format for the `Refine_Agent`.
    - **`Output_Scoring`:** Defines the expected output format for the `Output_Agent`.
- **`suggguest_outline.py`:** Contains the logic for generating instructions based on the question and desired score. This file uses a Google Gemini LLM to generate the instruction.
- **`requirements.txt`:** Lists the required Python packages.

### How it Works

The assistant uses a combination of LLMs from Groq and Google Gemini to perform different tasks:

- **Scoring:** The `model.py` file uses a Groq LLM to score essays based on the four IELTS criteria. The `Scoring_Agent` uses a prompt that describes the criteria and its sub-criteria, and the LLM generates a score for each sub-criteria and an overall score for the criteria.
- **Validation:** The `model.py` file uses a Groq LLM to provide detailed feedback on essays, including suggestions for improvement. The `Feedback_Agent` uses a prompt that describes the criteria and the essay, and the LLM generates feedback and improvement suggestions based on the essay's strengths and weaknesses.
- **Instruction:** The `suggguest_outline.py` file uses a Google Gemini LLM to generate instructions for writing essays based on the question and desired score. The LLM uses a prompt that describes the task requirements, essay types, essay structure, planning tips, and key skills. It then generates a detailed instruction that helps users plan and write their essays.

### Notes

- The assistant is still under development and may not always provide perfect results.
- The performance of the assistant depends on the quality of the LLMs used.
- The assistant is designed to help users improve their writing skills, not to provide complete answers.

### Contributing

Contributions are welcome!

### License

This project is licensed under the MIT License.
