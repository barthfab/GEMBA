import ipdb
import pandas as pd
import diskcache as dc
from gemba.gpt_api import GptApi, BatchGptApi
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from gemba.gemba_esa import TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING
from gemba.prompt import prompts, validate_number


def get_gemba_scores(
    source,
    hypothesis,
    source_lang,
    target_lang,
    method,
    model,
    load_batch=False,
    batch_ids=None,
    dataset_name=None,
    download_batches=False,
    local_path="",
):
    df = pd.DataFrame({"source_seg": source, "target_seg": hypothesis})
    df["source_lang"] = source_lang
    df["target_lang"] = target_lang

    cache = dc.Cache(
        f"cache/{model}_{method}",
        expire=None,
        size_limit=int(10e10),
        cull_limit=0,
        eviction_policy="none",
    )
    if load_batch or process_batch_index:
        gptapi = BatchGptApi(local_path=local_path)
    else:
        gptapi = GptApi()

    if method == "GEMBA-MQM":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        parse_answer = lambda x: parse_mqm_answer(
            x, list_mqm_errors=False, full_desc=True
        )
        answers = gptapi.bulk_request(
            df, model, parse_answer, cache=cache, max_tokens=500
        )
    elif method in [
        "GEMBA-DA",
        "GEMBA-DA_ref",
        "GEMBA-SQM",
        "GEMBA-SQM_ref",
        "GEMBA-stars",
        "GEMBA-stars_ref",
        "GEMBA-classes",
        "GEMBA-classes_ref",
    ]:
        df["prompt"] = df.apply(
            lambda x: apply_template(prompts[method]["prompt"], x), axis=1
        )
        parse_answer = prompts[method]["validate_answer"]
        answers = gptapi.bulk_request(
            df, model, parse_answer, cache=cache, max_tokens=500
        )
    elif method == "GEMBA-ESA":
        df["prompt"] = df.apply(
            lambda x: apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, x), axis=1
        )
        parse_answer = lambda x: x
        error_spans = gptapi.bulk_request(df, model, parse_answer, cache=cache)
        df["error_spans"] = pd.DataFrame(error_spans)["answer"]

        df["prompt"] = df.apply(
            lambda x: apply_template(TEMPLATE_GEMBA_ESA_RANKING, x), axis=1
        )
        parse_answer = validate_number
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache)
    elif method == "GEMBA-BATCH":
        if load_batch:
            df["prompt"] = df.apply(
                lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1
            )
            answer = gptapi.send_batches(df, model, dataset_name)
        elif process_batch_index:
            # TODO: load the data and process it
            df["prompt"] = df.apply(
                lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1
            )
            answer = gptapi.eval_results(batch_ids, dataset_name, download_batches)
        else:
            print(
                "You did not define if you want to process or load the batch. Thats why I'm not doing anything :)"
            )
            answer = None
        return [answer]
    else:
        raise Exception(f"Method {method} not supported.")

    return list(pd.DataFrame(answers)["answer"])
