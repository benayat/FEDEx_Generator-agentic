from fedex_generator.agents import utils
from fedex_generator.agents.code_combiner_agent import CodeCombinerAgent
from fedex_generator.agents.code_instructor_agent import CodeInstructionsAgent
from fedex_generator.agents.explanation_agent import ExplanationAgent
from fedex_generator.agents.matplotlib_code_agent import MatplotlibCodeAgent
from fedex_generator.agents.matplotlib_refine_agent import MatplotlibRefineAgent
import os
import asyncio

from fedex_generator.agents.single_agent import SingleAgent

base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
api_key = os.getenv("OPENAI_API_KEY", "ollama")

class AgentsManager:
    def __init__(self, base_url_local=base_url, api_key_local=api_key):
        """
        Initializes all the agents (ExplanationAgent, CodeInstructorAgent, MatplotlibCodeAgent, MatplotlibRefineAgent)
        and manages their activity.
        """
        explanation_model = os.getenv("EXPLANATION_MODEL", "gpt-4o")
        code_instructor_model = os.getenv("CODE_INSTRUCTOR_MODEL", "gpt-4o")
        matplotlib_code_model = os.getenv("MATPLOTLIB_CODE_MODEL", "gpt-4o")
        matplotlib_refine_model = os.getenv("MATPLOTLIB_REFINE_MODEL", "gpt-4o")
        code_combiner_model = os.getenv("CODE_COMBINER_MODEL", "gpt-4o")

        # Initialize the agents
        self.explanation_agent = ExplanationAgent(base_url_local, api_key_local, explanation_model)
        self.code_instructor_agent = CodeInstructionsAgent(base_url_local, api_key_local, code_instructor_model)
        self.matplotlib_code_agent = MatplotlibCodeAgent(base_url_local, api_key_local, matplotlib_code_model)
        self.matplotlib_refine_agent = MatplotlibRefineAgent(base_url_local, api_key_local, matplotlib_refine_model)
        self.code_combiner_agent = CodeCombinerAgent(base_url_local, api_key_local, code_combiner_model)
        self.single_agent = SingleAgent(base_url_local, api_key_local, 'gpt-4o')
    async def generate_explanation(self, analysis_result):
        """
        Generates a rephrased explanation using the ExplanationAgent.

        Parameters:
        analysis_result (dict): Analysis result dictionary.

        Returns:
        str: Rephrased explanation.
        """
        return await self.explanation_agent.generate_explanation(analysis_result)

    async def generate_code_instructions(self, explanation, data_dict):
        """
        Generates code instructions using the CodeInstructionsAgent.

        Parameters:
        explanation (str): Explanation of the data and visualization.
        data_dict (dict): Dictionary containing relevant data for the code instructions.

        Returns:
        str: Code instructions.
        """
        return await self.code_instructor_agent.generate_code_instructions(explanation, data_dict)

    async def generate_matplotlib_code(self, code_instructions):
        """
        Generates Matplotlib code using the MatplotlibCodeAgent.

        Parameters:
        code_instructions (str): Code instructions for generating Matplotlib plots.

        Returns:
        str: Matplotlib code.
        """
        return await self.matplotlib_code_agent.generate_matplotlib_code(code_instructions)

    async def refine_matplotlib_code(self, generated_code):
        """
        Refines the Matplotlib code to improve its layout using the MatplotlibRefineAgent.

        Parameters:
        generated_code (str): The generated Matplotlib code to refine.

        Returns:
        str: Refined Matplotlib code.
        """
        return await self.matplotlib_refine_agent.refine_matplotlib_code(generated_code)

    async def run_pipline_for_single_plot(self, analysis_result):
        """
        Manages the full process from generating the explanation to refining the Matplotlib code.

        Parameters:
        analysis_result (dict): Analysis result data for the ExplanationAgent.
        data_dict (dict): Data dictionary for the CodeInstructionsAgent.

        Returns:
        str: Final refined Matplotlib code.
        """
        # Step 1: Generate a rephrased explanation
        explanation, data_dict = await self.generate_explanation(analysis_result)

        # Step 2: Generate detailed code instructions based on the explanation and data
        code_instructions = await self.generate_code_instructions(explanation, data_dict)

        # Step 3: Generate Matplotlib code based on the instructions
        matplotlib_code = await self.generate_matplotlib_code(code_instructions)

        # Step 4: Refine the generated Matplotlib code for better layout and visual appeal
        refined_code = await self.refine_matplotlib_code(matplotlib_code)
        return refined_code if refined_code else matplotlib_code


    async def run_single_agent(self, analysis_result):
        """
        runs a single agent with improved prompt, and returns a json object with both explanation and code snippet.
        example: {
  "rephrased_explanation": "We found that people's age affects how much they spend.",
  "matplotlib_code": "
    import matplotlib.pyplot as plt

    age_groups = ['25', '30', '35']
    spending = [1000, 1500, 1200]

    plt.bar(age_groups, spending)
    plt.title('Age Group vs. Spending')
    plt.xlabel('Age Group')
    plt.ylabel('Average Spending')
    plt.show()
  "
}
        """
        return await self.single_agent.generate_explanation_and_code_in_json(analysis_result)
    async def run_single_agents_concurrently(self, analysis_results):
        """
        Manages the full process for multiple plots concurrently.
        generates a json list of dicts of explanation and code snippets for each plot, and runs it concurrently.

        Parameters:
        analysis_results (List[dict]): List of analysis result data for the ExplanationAgent.

        Returns:
        List[dict]: List of json objects with rephrased explanations and Matplotlib code
        """
        # Run the pipeline concurrently for each plot
        explanation_code_list = await asyncio.gather(*[self.run_single_agent(analysis_result) for analysis_result in analysis_results])
        for explanation, code in explanation_code_list:
            print(explanation)
            utils.execute_generated_code(code)


    async def run_single_agents_serially(self, analysis_results):
        """
        Manages the full process for multiple plots serially.
        generates a json list of dicts of explanation and code snippets for each plot, and runs it serially.

        Parameters:
        analysis_results (List[dict]): List of analysis result data for the ExplanationAgent.

        Returns:
        List[dict]: List of json objects with rephrased explanations and Matplotlib code
        """
        # explanation_code_list = []
        for analysis_result in analysis_results:
            explanation, code = await self.run_single_agent(analysis_result)
            utils.execute_generated_code(code, debug=True)
    async def run_pipelines_for_multiple_plots_concurrently(self, analysis_results):
        """
        Manages the full process for multiple plots concurrently.

        Parameters:
        analysis_results (List[dict]): List of analysis result data for the ExplanationAgent.

        Returns:
        List[str]: List of final refined Matplotlib codes for each plot.
        """
        # Run the pipeline concurrently for each plot
        refined_codes = await asyncio.gather(*[self.run_pipline_for_single_plot(analysis_result) for analysis_result in analysis_results])
        refined_codes = [str(code) for code in refined_codes]
        final_code = await self.code_combiner_agent.combine_multiple_code_strings(refined_codes) if len(refined_codes)>1 else refined_codes[0]
        utils.execute_generated_code(final_code)