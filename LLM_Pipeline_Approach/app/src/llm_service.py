import os
import yaml
import json
import re
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

model4_name = config["model4_name"]
model3_name = config["model3_name"]
api_key = config["openai_api_key"]


def parse_llm_response(llm_answer):
    """
    Parse LLM response and extract JSON, handling markdown code blocks.
    Returns the parsed JSON object.
    """
    content = llm_answer.content
    
    # Remove markdown code blocks if present
    if '```json' in content:
        match = re.search(r'```json\s*\n(.*?)```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = content
    elif '```' in content:
        match = re.search(r'```\s*\n(.*?)```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = content
    else:
        json_str = content
    
    return json.loads(json_str.strip())


def store_reasoning(key, reasoning_data):
    """
    Store LLM reasoning in session state for later display.
    
    Args:
        key: Unique identifier for this decision (e.g., 'target_selection')
        reasoning_data: Dict or string containing reasoning
    """
    if 'llm_reasoning' not in st.session_state:
        st.session_state.llm_reasoning = {}
    st.session_state.llm_reasoning[key] = reasoning_data


def decide_encode_type(attributes, data_frame_head, model_type=4, user_api_key=None):
    """
    Decides the encoding type for given attributes using a language model via the OpenAI API.

    Parameters:
    - attributes (list): A list of attributes for which to decide the encoding type.
    - data_frame_head (DataFrame): The head of the DataFrame containing the attributes. This parameter is expected to be a representation of the DataFrame (e.g., a string or a small subset of the actual DataFrame) that gives an overview of the data.
    - model_type (int, optional): Specifies the model to use. The default model_type=4 corresponds to a predefined model named `model4_name`. Another option is model_type=3, which corresponds to `model3_name`.
    - user_api_key (str, optional): The user's OpenAI API key. If not provided, a default API key `api_key` is used.

    Returns:
    - A JSON object containing the recommended encoding types for the given attributes. Please refer to prompt templates in config.py for details.
    
    Side effect: Stores reasoning in st.session_state

    Raises:
    - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
    """
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)
        
        template = config["numeric_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "data_frame_head"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, data_frame_head=data_frame_head)
        
        llm_answer = llm([HumanMessage(content=summary_prompt)])
        response = parse_llm_response(llm_answer)
        
        # Store reasoning
        store_reasoning('encoding', response.get('reasoning', {}))
        
        return response['decisions']
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.error(f"Error details: {str(e)}")
        st.stop()


def decide_fill_null(attributes, types_info, description_info, model_type=4, user_api_key=None):
    """
    Decides the best encoding type for given attributes using an AI model via OpenAI API.

    Parameters:
    - attributes (list): List of attribute names to consider for encoding.
    - data_frame_head (DataFrame or str): The head of the DataFrame or a string representation, providing context for the encoding decision.
    - model_type (int, optional): The model to use, where 4 is the default. Can be customized to use a different model.
    - user_api_key (str, optional): The user's OpenAI API key. If None, a default key is used.

    Returns:
    - dict: A JSON object with recommended encoding types for the attributes. Please refer to prompt templates in config.py for details.
    
    Side effect: Stores reasoning in st.session_state

    Raises:
    - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
    """
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)
        
        template = config["null_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "types_info", "description_info"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, types_info=types_info, description_info=description_info)
        
        llm_answer = llm([HumanMessage(content=summary_prompt)])
        response = parse_llm_response(llm_answer)
        
        # Store reasoning
        store_reasoning('null_filling', response.get('reasoning', {}))
        
        return response['decisions']
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.error(f"Error details: {str(e)}")
        st.stop()


def decide_model(shape_info, head_info, nunique_info, description_info, model_type=4, user_api_key=None):
    """
    Decides the most suitable machine learning model based on dataset characteristics.

    Parameters:
    - shape_info (dict): Information about the shape of the dataset.
    - head_info (str or DataFrame): The head of the dataset or its string representation.
    - nunique_info (dict): Information about the uniqueness of dataset attributes.
    - description_info (str): Descriptive information about the dataset.
    - model_type (int, optional): Specifies which model to consult for decision-making.
    - user_api_key (str, optional): OpenAI API key for making requests.

    Returns:
    - dict: A JSON object containing the recommended model and configuration. Please refer to prompt templates in config.py for details.
    
    Side effect: Stores reasoning in st.session_state

    Raises:
    - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
    """
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_model_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "head_info", "nunique_info", "description_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, head_info=head_info, nunique_info=nunique_info, description_info=description_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        response = parse_llm_response(llm_answer)
        
        # Store reasoning with overall strategy
        store_reasoning('model_selection', {
            'reasoning': response.get('reasoning', {}),
            'overall': response.get('overall_reasoning', '')
        })
        
        return response['decisions']
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.error(f"Error details: {str(e)}")
        st.stop()


def decide_cluster_model(shape_info, description_info, cluster_info, model_type=4, user_api_key=None):
    """
    Determines the appropriate clustering model based on dataset characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - cluster_info: Additional information relevant to clustering.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - A JSON object with the recommended clustering model and parameters. Please refer to prompt templates in config.py for details.
    
    Side effect: Stores reasoning in st.session_state

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_clustering_model_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "cluster_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, cluster_info=cluster_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        response = parse_llm_response(llm_answer)
        
        # Store reasoning
        store_reasoning('cluster_model_selection', {
            'reasoning': response.get('reasoning', {}),
            'overall': response.get('overall_reasoning', '')
        })
        
        return response['decisions']
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.error(f"Error details: {str(e)}")
        st.stop()


def decide_regression_model(shape_info, description_info, Y_name, model_type=4, user_api_key=None):
    """
    Determines the appropriate regression model based on dataset characteristics and the target variable.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - Y_name: The name of the target variable.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - A JSON object with the recommended regression model and parameters. Please refer to prompt templates in config.py for details.
    
    Side effect: Stores reasoning in st.session_state

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_regression_model_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "Y_name"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, Y_name=Y_name)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        response = parse_llm_response(llm_answer)
        
        # Store reasoning
        store_reasoning('regression_model_selection', {
            'reasoning': response.get('reasoning', {}),
            'overall': response.get('overall_reasoning', '')
        })
        
        return response['decisions']
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.error(f"Error details: {str(e)}")
        st.stop()


def decide_target_attribute(attributes, types_info, head_info, model_type=4, user_api_key=None):
    """
    Determines the target attribute for modeling based on dataset attributes and characteristics.

    Parameters:
    - attributes: A list of dataset attributes.
    - types_info: Information about the data types of the attributes.
    - head_info: A snapshot of the dataset's first few rows.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - The name of the recommended target attribute. Please refer to prompt templates in config.py for details.
    
    Side effect: Stores reasoning in st.session_state

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_target_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "types_info", "head_info"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, types_info=types_info, head_info=head_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        response = parse_llm_response(llm_answer)
        
        # Store reasoning
        store_reasoning('target_selection', response.get('reasoning', 'No reasoning provided'))
        
        return response['decision']['target']
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.error(f"Error details: {str(e)}")
        st.stop()


def decide_test_ratio(shape_info, model_type=4, user_api_key=None):
    """
    Determines the appropriate train-test split ratio based on dataset characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - The recommended train-test split ratio as a float. Please refer to prompt templates in config.py for details.
    
    Side effect: Stores reasoning in st.session_state

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_test_ratio_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        response = parse_llm_response(llm_answer)
        
        # Store reasoning
        store_reasoning('test_ratio', response.get('reasoning', 'No reasoning provided'))
        
        return response['decision']['test_ratio']
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.error(f"Error details: {str(e)}")
        st.stop()


def decide_balance(shape_info, description_info, balance_info, model_type=4, user_api_key=None):
    """
    Determines the appropriate method to balance the dataset based on its characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - balance_info: Additional information relevant to dataset balancing.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - The recommended method to balance the dataset. Please refer to prompt templates in config.py for details.
    
    Side effect: Stores reasoning in st.session_state

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_balance_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "balance_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, balance_info=balance_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        response = parse_llm_response(llm_answer)
        
        # Store reasoning
        store_reasoning('balance_strategy', response.get('reasoning', 'No reasoning provided'))
        
        return response['decision']['method']
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.error(f"Error details: {str(e)}")
        st.stop()
