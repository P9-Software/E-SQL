import os
from utils.prompt_utils import *
from utils.db_utils import * 
from utils.pipeline_utils import create_response
from typing import Dict

class Pipeline():
    def __init__(self, transformers_pipe, db_path):

        # Question Enrichment Arguments
        self.enrichment_level = "complex"
        self.elsn = 3
        self.efsse = False

        # Schema Filtering Arguments
        self.flsn = 3
        self.ffsse = False

        # SQL Generation Arguments
        self.cfg = True
        self.glsn = 3
        self.gfsse = False

        # Database Sample Arguments
        self.db_sample_limit = 5
        self.db_path = db_path

        # Relevant Description Arguments
        self.rdn = 6

        # Miscellaneous Arguments
        self.seed = 42

        # Pipeline attribute
        self.transformers_pipe = transformers_pipe

    def forward_pipeline_SF_CSG_QE_SR(self, t2s_object: Dict) -> Dict:
        """
        The function performs, Schema Filtering(SF) Candidate SQL Generation(CSG), Quesiton Enrichment(QE) and SQL Refinement(SR) modules respectively without filtering stages.
        
        Arguments:
            t2s_object (Dict): Python dictionary that stores information about a question like q_id, db_id, question, evidence etc. 
        Returns:
            t2s_object_prediction (Dict):  Python dictionary that stores information about a question like q_id, db_id, question, evidence etc and also stores information after each stage
        """
        db_id = t2s_object["db_id"]
        q_id = t2s_object["question_id"]
        evidence = t2s_object["evidence"]
        question = t2s_object["question"]
        
        database_column_meaning_path = os.path.join(os.path.dirname("self.db_path"), "column_descriptions.json")
        db_column_meanings = db_column_meaning_prep(database_column_meaning_path, db_id)
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # extracting original schema dictionary 
        original_schema_dict = get_schema_tables_and_columns_dict(db_path=self.db_path)


        ### STAGE 1: FILTERING THE DATABASE SCHEMA
        # -- original question is used.
        # -- Original Schema is used.
        schema_filtering_response_obj = self.schema_filtering_module(db_path=self.db_path, db_id=db_id, question=question, evidence=evidence, schema_dict=original_schema_dict, db_descriptions=db_descriptions)
        # print("schema_filtering_response_obj: \n", schema_filtering_response_obj)
        try:
            t2s_object["schema_filtering"] = {
                "filtering_reasoning": schema_filtering_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "filtered_schema_dict": schema_filtering_response_obj.choices[0].message.content['tables_and_columns'],
            }
        except Exception as e:
            logging.error(f"Error in reaching content from schema filtering response for question_id {q_id}: {e}")
            t2s_object["schema_filtering"] = f"{e}"
            return t2s_object

        ### STAGE 1.1: FILTERED SCHEMA CORRECTION
        filtered_schema_dict = schema_filtering_response_obj.choices[0].message.content['tables_and_columns']
        filtered_schema_dict, filtered_schema_problems = filtered_schema_correction(db_path=self.db_path, filtered_schema_dict=filtered_schema_dict) 
        t2s_object["schema_filtering_correction"] = {
            "filtered_schema_problems": filtered_schema_problems,
            "final_filtered_schema_dict": filtered_schema_dict
        }

        schema_statement = generate_schema_from_schema_dict(db_path=self.db_path, schema_dict=filtered_schema_dict)
        t2s_object["create_table_statement"] = schema_statement

        ### STAGE 2: Candidate SQL GENERATION
        # -- Original question is used
        # -- Filtered Schema is used 
        sql_generation_response_obj =  self.candidate_sql_generation_module(db_path=self.db_path, db_id=db_id, question=question, evidence=evidence, filtered_schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
        try:
            possible_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "possible_sql": possible_sql,
                "exec_err": "",
            }
            t2s_object["possible_sql"] = possible_sql
            # execute SQL
            try:
                _ = func_timeout(30, execute_sql, args=(self.db_path, possible_sql))
            except FunctionTimedOut:
                t2s_object['candidate_sql_generation']["exec_err"] = "timeout"
            except Exception as e:
                t2s_object['candidate_sql_generation']["exec_err"] = str(e)
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["candidate_sql_generation"] = {
                "sql_generation_reasoning": "",
                "possible_sql": "",
            }
            t2s_object["candidate_sql_generation"]["error"] = f"{e}"
            return t2s_object
        
        # Extract possible conditions dict list
        possible_conditions_dict_list = collect_possible_conditions(db_path=self.db_path, sql=possible_sql)
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)

        ### STAGE 3: Question Enrichment:
        # -- Original question is used
        # -- Original schema is used
        # -- Possible conditions are used
        q_enrich_response_obj = self.question_enrichment_module(db_path=self.db_path, q_id=q_id, db_id=db_id, question=question, evidence=evidence, possible_conditions=possible_conditions, schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
        try:
            enriched_question = q_enrich_response_obj.choices[0].message.content['enriched_question']
            enrichment_reasoning = q_enrich_response_obj.choices[0].message.content['chain_of_thought_reasoning']
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": q_enrich_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "enriched_question": q_enrich_response_obj.choices[0].message.content['enriched_question'],
            }
            enriched_question = question + enrichment_reasoning + enriched_question # This is added after experiment-24
        except Exception as e:
            logging.error(f"Error in reaching content from question enrichment response for question_id {q_id}: {e}")
            t2s_object["question_enrichment"] = {
                "enrichment_reasoning": "",
                "enriched_question": "",
            }
            t2s_object["question_enrichment"]["error"] = f"{e}"
            enriched_question = question
        
        ### STAGE 4: SQL Refinement
        # -- Enriched question is used
        # -- Original Schema is used 
        # -- Possible SQL is used
        # -- Possible Conditions is extracted from possible SQL and then used for augmentation
        # -- Execution Error for Possible SQL is used
        exec_err = t2s_object['candidate_sql_generation']["exec_err"]
        sql_generation_response_obj =  self.sql_refinement_module(db_path=self.db_path, db_id=db_id, question=enriched_question, evidence=evidence, possible_sql=possible_sql, exec_err=exec_err, filtered_schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
        try:
            predicted_sql = sql_generation_response_obj.choices[0].message.content['SQL']
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": sql_generation_response_obj.choices[0].message.content['chain_of_thought_reasoning'],
                "predicted_sql": predicted_sql,
            }
            t2s_object["predicted_sql"] = predicted_sql
        except Exception as e:
            logging.error(f"Error in reaching content from sql generation response for question_id {q_id}: {e}")
            t2s_object["sql_refinement"] = {
                "sql_generation_reasoning": "",
                "predicted_sql": "",
            }
            t2s_object["sql_refinement"]["error"] = f"{e}"
            return t2s_object

        t2s_object_prediction = t2s_object
        return t2s_object_prediction

        
    def construct_question_enrichment_prompt(self, db_path: str, q_id: int, db_id: str, question: str, evidence: str, possible_conditions: str, schema_dict: Dict, db_descriptions: str) -> str:
        """
        The function constructs the prompt required for the question enrichment stage

        Arguments:
            db_path (str): path to database sqlite file
            q_id (int): question id
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            possible_conditions (str): Possible conditions extracted from the previously generated possible SQL for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            prompt (str): Question enrichment prompt
        """
        enrichment_template_path = os.path.join(os.getcwd(), "prompt_templates/question_enrichment_prompt_template.txt")
        question_enrichment_prompt_template = extract_question_enrichment_prompt_template(enrichment_template_path)
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        q_enrich_few_shot_examples = question_enrichment_few_shot_prep(few_shot_data_path, q_id=q_id, q_db_id=db_id, level_shot_number=self.elsn, schema_existance=self.efsse, enrichment_level=self.enrichment_level, mode=self.mode)
        db_samples = extract_db_samples_enriched_bm25(question, evidence, db_path=db_path, schema_dict=schema_dict, sample_limit=self.db_sample_limit)
        schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=schema_dict)
        prompt = fill_question_enrichment_prompt_template(template=question_enrichment_prompt_template, schema=schema, db_samples=db_samples, question=question, possible_conditions=possible_conditions, few_shot_examples=q_enrich_few_shot_examples, evidence=evidence, db_descriptions=db_descriptions)
        # print("question_enrichment_prompt: \n", prompt)
        return prompt
    
    def question_enrichment_module(self, db_path: str, q_id: int, db_id: str, question: str, evidence: str, possible_conditions: str, schema_dict: Dict, db_descriptions: str) -> Dict:
        """
        The function enrich the given question using LLM.

        Arguments:
            db_path (str): path to database sqlite file
            q_id (int): question id
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            possible_conditions (str): possible conditions extracted from previously generated possible SQL for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        prompt = self.construct_question_enrichment_prompt(db_path=db_path, q_id=q_id, db_id=db_id, question=question, evidence=evidence, possible_conditions=possible_conditions, schema_dict=schema_dict, db_descriptions=db_descriptions)
        response_object = create_response(stage="question_enrichment", prompt=prompt, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=self.n)

        return response_object
    
    def construct_candidate_sql_generation_prompt(self, db_path: str, db_id: int, question: str, evidence: str, filtered_schema_dict: Dict, db_descriptions: str)->str:
        """
        The function constructs the prompt required for the candidate SQL generation stage.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            prompt (str): prompt for SQL generation stage
        """
        sql_generation_template_path =  os.path.join(os.getcwd(), "prompt_templates/candidate_sql_generation_prompt_template.txt")
        with open(sql_generation_template_path, 'r') as f:
            sql_generation_template = f.read()
            
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        sql_generation_few_shot_examples = sql_generation_and_refinement_few_shot_prep(few_shot_data_path, q_db_id=db_id, level_shot_number=self.glsn, schema_existance=self.gfsse, mode=self.mode)
        db_samples = extract_db_samples_enriched_bm25(question, evidence, db_path, schema_dict=filtered_schema_dict, sample_limit=self.db_sample_limit)
        filtered_schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=filtered_schema_dict)
        prompt = fill_candidate_sql_prompt_template(template=sql_generation_template, schema=filtered_schema, db_samples=db_samples, question=question, few_shot_examples=sql_generation_few_shot_examples, evidence=evidence, db_descriptions=db_descriptions) 
        # print("candidate_sql_prompt: \n", prompt)
        return prompt

    
    def construct_sql_refinement_prompt(self, db_path: str, db_id: int, question: str, evidence: str, possible_sql: str, exec_err: str, filtered_schema_dict: Dict, db_descriptions: str)->str:
        """
        The function constructs the prompt required for the SQL refinement stage.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            possible_sql (str): Previously generated possible SQL for the question
            exec_err (str): Taken execution error when possible SQL is executed
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            prompt (str): prompt for SQL generation stage
        """
        sql_generation_template_path =  os.path.join(os.getcwd(), "prompt_templates/sql_refinement_prompt_template.txt")
        with open(sql_generation_template_path, 'r') as f:
            sql_generation_template = f.read()
            
        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        sql_generation_few_shot_examples = sql_generation_and_refinement_few_shot_prep(few_shot_data_path, q_db_id=db_id, level_shot_number=self.glsn, schema_existance=self.gfsse, mode=self.mode)
        possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=possible_sql)
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)
        filtered_schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=filtered_schema_dict)
        prompt = fill_refinement_prompt_template(template=sql_generation_template, schema=filtered_schema, possible_conditions=possible_conditions, question=question, possible_sql=possible_sql, exec_err=exec_err, few_shot_examples=sql_generation_few_shot_examples, evidence=evidence, db_descriptions=db_descriptions) 
        # print("refinement_prompt: \n", prompt)
        return prompt
    
    def construct_filtering_prompt(self, db_path: str, db_id: str, question: str, evidence: str, schema_dict: Dict, db_descriptions: str)->str:
        """
        The function constructs the prompt required for the database schema filtering stage

        Arguments:  
            db_path (str): The database sqlite file path.
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            prompt (str): prompt for database schema filtering stage
        """
        schema_filtering_prompt_template_path =  os.path.join(os.getcwd(), "prompt_templates/schema_filter_prompt_template.txt")
        with open(schema_filtering_prompt_template_path, 'r') as f:
            schema_filtering_template = f.read()

        few_shot_data_path = os.path.join(os.getcwd(), "few-shot-data/question_enrichment_few_shot_examples.json")
        schema_filtering_few_shot_examples = schema_filtering_few_shot_prep(few_shot_data_path, q_db_id=db_id, level_shot_number=self.elsn, schema_existance=self.efsse, mode=self.mode)
        db_samples = extract_db_samples_enriched_bm25(question, evidence, db_path=db_path, schema_dict=schema_dict, sample_limit=self.db_sample_limit)
        schema = generate_schema_from_schema_dict(db_path=db_path, schema_dict=schema_dict)
        prompt = fill_prompt_template(template=schema_filtering_template, schema=schema, db_samples=db_samples, question=question, few_shot_examples=schema_filtering_few_shot_examples, evidence=evidence, db_descriptions=db_descriptions)
        # print("\nSchema Filtering Prompt: \n", prompt)
    
        return prompt

    
    def candidate_sql_generation_module(self, db_path: str, db_id: int, question: str, evidence: str, filtered_schema_dict: Dict, db_descriptions: str):
        """
        This function generates candidate SQL for answering the question.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        prompt = self.construct_candidate_sql_generation_prompt(db_path=db_path, db_id=db_id, question=question, evidence=evidence, filtered_schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
        response_object = create_response(stage="candidate_sql_generation", prompt=prompt, pipe=self.transformers_pipe)

        return response_object

    
    def sql_refinement_module(self, db_path: str, db_id: int, question: str, evidence: str, possible_sql: str, exec_err: str, filtered_schema_dict: Dict, db_descriptions: str):
        """
        This function refines or re-generates a SQL query for answering the question.
        Possible SQL query, possible conditions generated from possible SQL query and execution error if it is exist are leveraged for better SQL refinement.

        Arguments:
            db_path (str): The database sqlite file path.
            db_id (int): database ID, i.e. database name
            question (str):  Natural language question 
            evidence (str): evidence for the question
            possible_sql (str): Previously generated possible SQL query for the question
            exec_err (str): Taken execution error when possible SQL is executed 
            filtered_schema_dict (Dict[str, List[str]]): filtered database as dictionary where keys are the tables and the values are the list of column names
            db_descriptions (str): Question relevant database item (column) descriptions
        
        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        prompt = self.construct_sql_refinement_prompt(db_path=db_path, db_id=db_id, question=question, evidence=evidence, possible_sql=possible_sql, exec_err=exec_err, filtered_schema_dict=filtered_schema_dict, db_descriptions=db_descriptions)
        response_object = create_response(stage="sql_refinement", prompt=prompt, pipe=self.pipeline)

        return response_object
    

    def schema_filtering_module(self, db_path: str, db_id: str, question: str, evidence: str, schema_dict: Dict, db_descriptions: str):
        """
        The function filters the database schema by eliminating the unnecessary tables and columns

        Arguments:  
            db_path (str): The database sqlite file path.
            db_id (str): database ID, i.e. database name
            question (str): Natural language question 
            evidence (str): evidence for the question
            schema_dict (Dict[str, List[str]]): database schema dictionary
            db_descriptions (str): Question relevant database item (column) descriptions

        Returns:
            response_object (Dict): Response object returned by the LLM
        """
        prompt = self.construct_filtering_prompt(db_path=db_path, db_id=db_id, question=question, evidence=evidence, schema_dict=schema_dict, db_descriptions=db_descriptions)
        response_object = create_response(stage="schema_filtering", prompt=prompt, pipe=self.pipeline)

        return response_object
    
    
