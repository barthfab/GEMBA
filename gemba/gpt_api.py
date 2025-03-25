import os
import sys
import time
import logging
from termcolor import colored
import openai
import tqdm
import json
import pandas as pd


class BatchGptApi:
    def __init__(
        self,
        verbose=False,
        model="gpt-4-turbo",
        temperature=0.7,
        output_dir="batch_requests",
        batch_size: int = 3,
        local_path: str = "./output",
    ):
        """
        Initialize the BatchGptApi class with predefined hyperparameters and model settings.

        :param api_key: OpenAI API key.
        :param model: Model to use for batch processing (default: gpt-4-turbo).
        :param temperature: Sampling temperature for response variability.
        :param output_dir: Directory to save JSONL batch files.
        """
        self.model = model
        self.temperature = temperature
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.local_path = local_path

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.verbose = verbose

        if "OPENAI_AZURE_ENDPOINT" in os.environ:
            assert (
                "OPENAI_AZURE_KEY" in os.environ
            ), "OPENAI_AZURE_KEY not found in environment"

            # Azure API access
            self.client = openai.AzureOpenAI(
                api_key=os.environ["OPENAI_AZURE_KEY"],
                azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                api_version="2023-07-01-preview",
            )
        elif "OPENAI_API_KEY" in os.environ:
            # OpenAI API access
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise Exception(
                "OPENAI_API_KEY or OPENAI_AZURE_KEY not found in environment"
            )

        logging.getLogger().setLevel(
            logging.CRITICAL
        )  # in order to suppress all these HTTP INFO log messages

    def send_batches(self, df, model, dataset_name):
        batch_index = self.create_batches(df, model, dataset_name)
        batch_ids = self.upload_batches(batch_index, dataset_name)

    def create_batches(self, df, model, dataset_name):
        file_index = 1
        total_words = 0

        for i, row in df.iterrows():
            prompt = row["prompt"]  # Extract text from the DataFrame
            total_words += self.count_words(prompt)
            filename = f"batch_{dataset_name}_{file_index}.jsonl"

            try:
                file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
            except OSError:
                file_size = 0

            # If file is too large or word count exceeded, create a new file
            if file_size > 200 or total_words > 50000:
                file_index += 1
                total_words = 0  # Reset word count

            # Write current batch to JSON
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    self.gen_json(f"batch_index_{file_index}_{i}", prompt, model),
                    f,
                )
                f.write("\n")

        return file_index

    def upload_batches(self, batch_index, dataset_name):
        batch_ids = []
        for batch in range(1, batch_index + 1):
            with open(f"batch_{dataset_name}_{batch}.jsonl", "rb") as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch",
                )
            batch_status = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_ids.append(batch_status.id)
        return batch_ids

    def count_words(self, prompt):
        words = 0
        for input in prompt:
            words += len(str(input["content"]).split())
        return words

    def gen_json(self, custom_id, prompt, model, temperature=0.7, max_tokens=100):
        return (
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "messages": prompt,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            },
        )

    def eval_results(self, batch_ids, dataset_name, download=True):
        if download:
            batch_path = self.download_batches(batch_ids)
        else:
            batch_path = self.local_path
        answers = self.extract_answers_from_batch(batch_path)
        return answers

    def download_batches(self, batch_ids):
        batch_outputs = []
        for batch_id in batch_ids:
            batch_info = self.client.batches.retrieve(batch_id)
            if batch_info.status == "completed":
                breakpoint()
                file_response = self.client.files.content(batch_info.output_file_id)
                batch_outputs.append(file_response.text)
                # TODO: download results and return teh path to the results jsons
            else:
                print(
                    f"batch {batch_id} is not completed. Current sstatus is {batch_info.status}"
                )
        return batch_outputs

    def extract_answers_from_batch(self, batch_path):
        # TODO: extract the generated answers and use the existing functions from GptApi to process the results
        pass


# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, verbose=False):
        self.verbose = verbose

        if "OPENAI_AZURE_ENDPOINT" in os.environ:
            assert (
                "OPENAI_AZURE_KEY" in os.environ
            ), "OPENAI_AZURE_KEY not found in environment"

            # Azure API access
            self.client = openai.AzureOpenAI(
                api_key=os.environ["OPENAI_AZURE_KEY"],
                azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                api_version="2023-07-01-preview",
            )
        elif "OPENAI_API_KEY" in os.environ:
            # OpenAI API access
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise Exception(
                "OPENAI_API_KEY or OPENAI_AZURE_KEY not found in environment"
            )

        logging.getLogger().setLevel(
            logging.CRITICAL
        )  # in order to suppress all these HTTP INFO log messages

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(
        self,
        prompt,
        model,
        parse_response,
        temperature=0,
        answer_id=-1,
        cache=None,
        max_tokens=None,
    ):
        request = {"model": model, "temperature": temperature, "prompt": prompt}

        if request in cache and cache[request] is not None and len(cache[request]) > 0:
            answers = cache[request]
        else:
            answers = self.request_api(prompt, model, temperature, max_tokens)
            cache[request] = answers

        # there is no valid answer
        if len(answers) == 0:
            return [
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": None,
                    "model": model,
                }
            ]

        parsed_answers = []
        for full_answer in answers:
            finish_reason = full_answer["finish_reason"]
            full_answer = full_answer["answer"]
            answer_id += 1
            answer = parse_response(full_answer)
            if self.verbose or temperature > 0:
                print(
                    f"Answer (t={temperature}): "
                    + colored(answer, "yellow")
                    + " ("
                    + colored(full_answer, "blue")
                    + ")",
                    file=sys.stderr,
                )
            if answer is None:
                continue
            parsed_answers.append(
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": answer,
                    "prompt": prompt,
                    "finish_reason": finish_reason,
                    "model": model,
                }
            )

        # there was no valid answer, increase temperature and try again
        if len(parsed_answers) == 0:
            return self.request(
                prompt,
                model,
                parse_response,
                temperature=temperature + 1,
                answer_id=answer_id,
                cache=cache,
            )

        return parsed_answers

    def request_api(self, prompt, model, temperature=0, max_tokens=None):
        if temperature > 10:
            return []

        while True:
            try:
                response = self.call_api(prompt, model, temperature, max_tokens)
                break
            except Exception as e:
                # response was filtered
                if hasattr(e, "code"):
                    if e.code == "content_filter":
                        return []
                    print(e.code, file=sys.stderr)
                if hasattr(e, "error") and e.error["code"] == "invalid_model_output":
                    return []

                # frequent error is reaching the API limit
                print(colored("Error, retrying...", "red"), file=sys.stderr)
                print(e, file=sys.stderr)
                time.sleep(1)

        answers = []
        for choice in response.choices:
            if choice.message.content is None:
                return []
            if hasattr(choice, "message"):
                answer = choice.message.content.strip()
            else:
                answer = choice.text.strip()

            # one of the responses didn't finish, we need to request more tokens
            if choice.finish_reason != "stop":
                if self.verbose:
                    print(
                        colored(f"Increasing max tokens to fit answers.", "red")
                        + colored(answer, "blue"),
                        file=sys.stderr,
                    )
                print(f"Finish reason: {choice.finish_reason}", file=sys.stderr)
                if max_tokens is None:
                    return []
                return self.request_api(
                    prompt, model, temperature=temperature, max_tokens=max_tokens + 200
                )

            answers.append(
                {
                    "answer": answer,
                    "finish_reason": choice.finish_reason,
                }
            )

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, temperature, max_tokens):
        parameters = {
            "temperature": temperature / 10,
            "top_p": 1,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "model": model,
        }

        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens

        if isinstance(prompt, list):
            # check that prompt contain list of dictionaries with role and content
            assert all(
                isinstance(p, dict) for p in prompt
            ), "Prompts must be a list of dictionaries."
            assert all(
                "role" in p and "content" in p for p in prompt
            ), "Prompts must be a list of dictionaries with role and content."

            parameters["messages"] = prompt
        else:
            parameters["messages"] = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        return self.client.chat.completions.create(**parameters)

    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=None):
        answers = []
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), file=sys.stderr):
            prompt = row["prompt"]
            parsed_answers = self.request(
                prompt, model, parse_mqm_answer, cache=cache, max_tokens=max_tokens
            )
            answers += parsed_answers
        return answers
