from openai import OpenAI, AsyncOpenAI
class BaseAgent(object):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.model = model
        self.sync_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate_chat_completion(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """
        Generates a chat completion using the specified model and prompts.

        Parameters:
        model (str): The model to be used for generating the completion.
        system_prompt (str): The system prompt to guide the model.
        user_prompt (str): The user prompt to guide the model.

        Returns:
        str: The generated completion content.
        """
        response = self.sync_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=temperature,
        )

        return response.choices[0].message.content

    async def generate_chat_completion_async(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """
        Generates a chat completion using the specified model and prompts asynchronously.

        Parameters:
        model (str): The model to be used for generating the completion.
        system_prompt (str): The system prompt to guide the model.
        user_prompt (str): The user prompt to guide the model.

        Returns:
        str: The generated completion content.
        """
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=temperature,
        )

        return response.choices[0].message.content