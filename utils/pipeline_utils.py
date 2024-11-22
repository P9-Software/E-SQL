from transformers import Pipeline
from typing import Dict

def create_response(stage: str, prompt: str, pipe: Pipeline) -> Dict:
    """
    The functions creates chat response by using chat completion

    Arguments:
        stage (str): stage in the pipeline
        prompt (str): prepared prompt 
        model (str): LLM model used to create chat completion
        max_tokens (int): The maximum number of tokens that can be generated in the chat completion

    Returns:
        response_object (Dict): Object returned by the model
    """

    if stage == "question_enrichment":
        system_content = "You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions and the evidence to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items."
    elif stage == "candidate_sql_generation":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "sql_refinement":
        system_content = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, evidence, possible SQL and possible conditions. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."
    elif stage == "schema_filtering":
        system_content = "You are an excellent data scientist. You can capture the link between a question and corresponding database and determine the useful database items (tables and columns) perfectly. Your objective is to analyze and understand the essence of the given question, corresponding database schema, database column descriptions, samples and evidence and then select the useful database items such as tables and columns. This database item filtering is essential for eliminating unnecessary information in the database so that corresponding structured query language (SQL) of the question can be generated correctly in later steps."
    else:
        raise ValueError("Wrong value for stage. It can only take following values: question_enrichment, candidate_sql_generation, sql_refinement or schema_filtering.")

    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
    response_object = pipe(messages)

    return response_object
