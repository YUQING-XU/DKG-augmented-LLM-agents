import copy
import sys
import os
#sys.path.append("../")
import json
import requests
import openai
from openai import OpenAI
import time
import random
from config import MAX_REFERENCE_NUM
import argparse
import pandas as pd
import json
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus


class OpenAI_model:
    def __init__(self, api_key: str, api_name: str,
                 get_reasoning_content: bool = False):
        self.api_key = api_key
        self.api_name = api_name
        self.get_reasoning_content = get_reasoning_content
        if(api_name == 'deepseek' or api_name == 'deepseek-v3'):
            ### using deepseek r1 model from ARK API
            print('using ARK API to respond via deepseek r1...')
            self.client = OpenAI(
                api_key = self.api_key,
                base_url = "https://ark.cn-beijing.volces.com/api/v3",
            )
        else:
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.oneabc.org/v1")

    def compeletion(self, model: str, messages: list, max_retries: int, **kwargs):
        retries = 0
        ret = {
            'content': '',
            'reasoning': ''
        }
        while retries < max_retries:
            try:
                if(self.api_name == 'deepseek'):
                    print('using deepseek model...')
                    model = 'ep-20250208151949-2c29b'
                elif(self.api_name == 'deepseek-v3'):
                    print('using deepseek v3 model...')
                    model= "deepseek-v3-250324"
                response = self.client.chat.completions.create(
                    model = model,
                    messages=messages,
                    **kwargs,
                    #reasoning="auto",
                )
                msg = response.choices[0].message.content
                assert isinstance(msg, str), "The retruned response is not a string."
                ret['content'] = msg
                if self.get_reasoning_content:
                    ret['reasoning'] = response.choices[0].message.reasoning_content
                return ret  # Return the response if successful

            except Exception as e:
                # Catch all other exceptions
                print(f"Unexpected error: {e}. Retrying in 5 seconds...")
                retries += 1
                time.sleep(1)
        
        return ''  # Return an empty string if max_retries is exceeded

class AgentAction:
    def __init__(self, chatbot='', 
                 max_new_tokens=1024,
                 api_name =None,
                 api_model = 'gpt-4o-mini',
                 api_token = None,
                 max_retry = 5,
                 temperature = 0.2,
                 get_reasoning_content = False,
                 **kwargs
                 ):
        '''
        api_name: str, the name of the api (use OpenAI API), if api is empty, use chatbot to respond
        '''
        self.api_token = api_token
        self.api_name = api_name
        self.api_model = api_model
        self.max_retry = max_retry
        self.temperature = temperature
        self.get_reasoning_content = get_reasoning_content

        if(not api_name):
            print('using HF chatbot to respond...')
            self.chatbot = chatbot
        else:
            print('using OpenAI API to respond...')
            self.chatbot = OpenAI_model(api_key=self.api_token, api_name = self.api_name, get_reasoning_content=self.get_reasoning_content)
        self.max_new_tokens = max_new_tokens


    def complete(self, prompt,**kwargs):

        ### use api
        
        # message_list = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": message}
        # ]
        message_list = [
            {"role": "user", "content": prompt}
        ]
        
        response_dict = self.chatbot.compeletion(self.api_model, message_list, self.max_retry, temperature = self.temperature, max_tokens = self.max_new_tokens)
        response = response_dict['content']
        if self.get_reasoning_content:
            reasoning = response_dict['reasoning']
            
        ## TODO : complete the parsing
        return response



def log(message, save_path):
    ### get dir
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(save_path):
        with open(save_path, 'w') as file:
            file.write(message + '\n')
    else:
        with open(save_path, 'a') as file:
            file.write(message + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="ductwork/ifc_file", help="Directory containing input IFC txt files")
    # deepseek or gpt-4o-mini 
    # parser.add_argument("--api_name", type=str, required=False, default='deepseek-v3')
    # parser.add_argument("--api_model", type=str, required=False, default='deepseek-v3')
    parser.add_argument("--api_name", type=str, required=False, default='gpt-4o')
    parser.add_argument("--api_model", type=str, required=False, default='gpt-4o')
    parser.add_argument("--api_token", type=str)
    parser.add_argument("--max_retry", type=int, required=False, default=5)
    parser.add_argument("--temperature", type=float, required=False, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, required=False, default=1024)
    args = parser.parse_args()

    # # Get list of input files
    # input_files = process_directory(args.input_dir)
    # if not input_files:
    #     print(f"No txt files found in {args.input_dir}")
    #     sys.exit(1)

    # Set up the agent
    if args.api_name == 'deepseek' or args.api_name == 'deepseek-v3':
        args.api_token = 'xxx'
    else:
        args.api_token = 'xxx'
    agent = AgentAction(api_name=args.api_name, api_model=args.api_model, api_token=args.api_token, 
                       max_retry=args.max_retry, temperature=args.temperature, max_new_tokens=args.max_new_tokens)



from openai import AzureOpenAI
client_azure = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="xxx",
    api_key="xxx"
)
client = client_azure
# Provide the model deployment name you want to use for this example
deployment_name = args.api_model

# 1. Load Inputs
# **(1) Load catalogs**
catalogs = {
    "air_diffuser": pd.read_excel("DKG_aug_LLMagents/catalogs/Diffuser_Catalog.xlsx", sheet_name="Sheet1"),
    "duct_segment": pd.read_excel("DKG_aug_LLMagents/catalogs/Duct_Catalog.xlsx", sheet_name="Sheet1"),
    "duct_fitting": pd.read_excel("DKG_aug_LLMagents/catalogs/Duct_Fitting_Catalog.xlsx", sheet_name="Sheet1"),
}

# **(2) Load current design**
with open("knowledge_graph_case_study.json") as f:
    current_design = json.load(f)

# 2. Calling Functions
# **(1) Current Carbon & Cost Calculation**
def get_current_stats(design: str) -> dict:
    # Load the design data first
    try:
        design_path = os.path.join('DKG_aug_LLMagents', design)
        with open(design_path, 'r') as f:
            design_data = json.load(f)
    except Exception as e:
        print(f"Error loading design data: {e}")
        return "Error: Could not load design data"
    
    design = design_data
    total_carbon = 0
    total_cost = 0
    air_terminal_carbon = 0
    air_terminal_cost = 0
    duct_segment_carbon = 0
    duct_segment_cost = 0
    duct_fitting_carbon = 0
    duct_fitting_cost = 0
    

    def traverse(node):
        nonlocal total_carbon, total_cost
        if isinstance(node, dict):
            if "properties" in node and isinstance(node["properties"], dict):
                props = node["properties"]
                if "Emission" in props:
                    total_carbon += float(props["Emission"])
                if "Cost" in props:
                    total_cost += float(props["Cost"])
            for value in node.values():
                traverse(value)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    def traverse_air_terminal(node):
        nonlocal air_terminal_carbon, air_terminal_cost
        if isinstance(node, dict):
            if "nodes" in node:
                for item in node["nodes"]:
                    if isinstance(item, dict) and item.get("type") == "AirTerminalProduct":
                        if "properties" in item and isinstance(item["properties"], dict):
                            props = item["properties"]
                            if "Emission" in props:
                                air_terminal_carbon += float(props["Emission"])
                            if "Cost" in props:
                                air_terminal_cost += float(props["Cost"])
            for value in node.values():
                traverse_air_terminal(value)
        elif isinstance(node, list):
            for item in node:
                traverse_air_terminal(item)
    
    def traverse_duct_segment(node):
        nonlocal duct_segment_carbon, duct_segment_cost
        if isinstance(node, dict):
            if "nodes" in node:
                for item in node["nodes"]:
                    if isinstance(item, dict) and item.get("type") == "DuctSegmentProduct":
                        if "properties" in item and isinstance(item["properties"], dict):
                            props = item["properties"]
                            if "Emission" in props:
                                duct_segment_carbon += float(props["Emission"])
                            if "Cost" in props:
                                duct_segment_cost += float(props["Cost"])
            for value in node.values():
                traverse_duct_segment(value)
        elif isinstance(node, list):
            for item in node:
                traverse_duct_segment(item)
    
    def traverse_duct_fitting(node):
        nonlocal duct_fitting_carbon, duct_fitting_cost
        if isinstance(node, dict):
            if "nodes" in node:
                for item in node["nodes"]:
                    if isinstance(item, dict) and item.get("type") == "DuctFittingProduct":
                        props = item.get("properties", {})
                        if isinstance(props, dict):
                            if "Emission" in props:
                                duct_fitting_carbon += float(props["Emission"])
                        if "Cost" in props:
                            duct_fitting_cost += float(props["Cost"])
            for value in node.values():
                traverse_duct_fitting(value)
        elif isinstance(node, list):
            for item in node:
                traverse_duct_fitting(item)
    
    traverse(design)
    traverse_air_terminal(design)
    traverse_duct_segment(design)
    traverse_duct_fitting(design)
    return {
        "total_carbon": total_carbon,
        "total_cost": total_cost,
        "air_terminal_carbon": air_terminal_carbon,
        "air_terminal_cost": air_terminal_cost,
        "duct_segment_carbon": duct_segment_carbon,
        "duct_segment_cost": duct_segment_cost,
        "duct_fitting_carbon": duct_fitting_carbon,
        "duct_fitting_cost": duct_fitting_cost
    }

# **(2) Find the lowest carbon combination**
def find_lowest_carbon_combination(
    json_file_path: str,
    diffuser_catalog: str,
    duct_segment_catalog: str,
    duct_fitting_catalog: str,
    json_size_mm: float,
    min_airflow_cfm: float,
    max_noise_nc: float,
    max_velocity_ms: float
) -> dict:
    
        # Load the design data first
    try:
        diffuser_catalog=pd.read_excel(f"DKG_aug_LLMagents/catalogs/{diffuser_catalog}", sheet_name="Sheet1")
        duct_segment_catalog=pd.read_excel(f"DKG_aug_LLMagents/catalogs/{duct_segment_catalog}", sheet_name="Sheet1")
        duct_fitting_catalog=pd.read_excel(f"DKG_aug_LLMagents/catalogs/{duct_fitting_catalog}", sheet_name="Sheet1")
        design_path = os.path.join('DKG_aug_LLMagents', json_file_path)
        with open(design_path, 'r') as f:
            design_data = json.load(f)
    except Exception as e:
        print(f"Error loading design data: {e}")
        return "Error: Could not load design data"
    
    current_design = design_data
    # Step 1: Filter catalogs based on requirements
    # Filter air diffusers
    filtered_diffuser_catalog = diffuser_catalog[
        (diffuser_catalog['ad_size_mm'] == json_size_mm) &
        (diffuser_catalog['ad_airflow_cfm'] >= min_airflow_cfm) &
        (diffuser_catalog['ad_nc'] <= max_noise_nc)
    ].copy()
    
    # Filter duct segments
    filtered_duct_segment_catalog = duct_segment_catalog[
        (duct_segment_catalog['ds_max_velocity_ms'] <= max_velocity_ms)
    ].copy()
    
    # # Step 2: Extract quantities from JSON
    # with open(json_file_path) as f:
    #     current_design = json.load(f)

    def get_quantity(design: dict) -> tuple:
        diffuser_count = 0
        duct_segment_volume = 0
        duct_fitting_volume = 0

        for node in design["nodes"]:
            if node["type"] == "AirTerminalProduct":
                diffuser_count += 1
            elif node["type"] == "DuctSegmentProduct":
                duct_segment_volume += float(node["properties"]["NetVolume"])
            elif node["type"] == "DuctFittingProduct":
                duct_fitting_volume += float(node["properties"]["NetVolume"])
            else:
                print(f"Unknown node type: {node['type']}")
        return diffuser_count, duct_segment_volume, duct_fitting_volume

    diffuser_count, duct_segment_volume, duct_fitting_volume = get_quantity(current_design)

    # Step 3: Preprocess catalogs (ensure manufacturer matching for ducts/fittings)
    manufacturers = set(filtered_duct_segment_catalog["ds_manufacturer"]).intersection(
        set(duct_fitting_catalog["df_manufacturer"])
    )
    filtered_duct_segment_catalog = filtered_duct_segment_catalog[
        filtered_duct_segment_catalog["ds_manufacturer"].isin(manufacturers)
    ]
    duct_fitting_catalog = duct_fitting_catalog[
        duct_fitting_catalog["df_manufacturer"].isin(manufacturers)
    ]      

    # Step 4: Create optimization problem
    prob = LpProblem("LowestCarbonCombination", LpMinimize)

    # Decision variables (binary: 1 if product is selected, 0 otherwise)
    diffuser_vars = LpVariable.dicts(
        "diffuser", filtered_diffuser_catalog["ad_id"], cat="Binary"
    )
    duct_segment_vars = LpVariable.dicts(
        "duct_segment", filtered_duct_segment_catalog["ds_id"], cat="Binary"
    )
    duct_fitting_vars = LpVariable.dicts(
        "duct_fitting", duct_fitting_catalog["df_id"], cat="Binary"
    )

    # Objective: Minimize total carbon
    prob += lpSum(
        # Diffuser carbon: count * emission_per_unit
        diffuser_count * row["ad_ec_punit"] * diffuser_vars[row['ad_id']]
        for index,row in filtered_diffuser_catalog.iterrows()
    ) + lpSum(
        # Duct segment carbon: volume * density * emission_per_kg
        duct_segment_volume * row["ds_density"] * 
        row["ds_ef_pkg"] * duct_segment_vars[row['ds_id']]
        for index,row in filtered_duct_segment_catalog.iterrows()
    ) + lpSum(
        # Duct fitting carbon: volume * density * emission_per_kg
        duct_fitting_volume * row["df_density_pm3"] * 
        row["df_ef_pkg"] * duct_fitting_vars[row['df_id']]
        for index,row in duct_fitting_catalog.iterrows()
    )

    # Constraints:
    # 1. Select exactly one diffuser, one duct segment, one duct fitting
    prob += lpSum(diffuser_vars[d] for d in filtered_diffuser_catalog["ad_id"]) == 1
    prob += lpSum(duct_segment_vars[ds] for ds in filtered_duct_segment_catalog["ds_id"]) == 1
    prob += lpSum(duct_fitting_vars[df] for df in duct_fitting_catalog["df_id"]) == 1

    # 2. Duct segment and fitting must have the same manufacturer
    for m in manufacturers:
        prob += lpSum(
            duct_segment_vars[ds] for ds in filtered_duct_segment_catalog[
                filtered_duct_segment_catalog["ds_manufacturer"] == m
            ]["ds_id"]
        ) == lpSum(
            duct_fitting_vars[df] for df in duct_fitting_catalog[
                duct_fitting_catalog["df_manufacturer"] == m
            ]["df_id"]
        )

    # Solve the problem
    prob.solve()

    # Step 5: Extract results
    selected_diffuser = next(
        d for d in filtered_diffuser_catalog["ad_id"] if diffuser_vars[d].value() == 1
    )
    selected_duct_segment = next(
        ds for ds in filtered_duct_segment_catalog["ds_id"] if duct_segment_vars[ds].value() == 1
    )
    selected_duct_fitting = next(
        df for df in duct_fitting_catalog["df_id"] if duct_fitting_vars[df].value() == 1
    )

    # Calculate total carbon and cost
    total_carbon = (
        diffuser_count * filtered_diffuser_catalog.loc[filtered_diffuser_catalog["ad_id"] == selected_diffuser, "ad_ec_punit"].iloc[0] +
        duct_segment_volume * filtered_duct_segment_catalog.loc[filtered_duct_segment_catalog["ds_id"] == selected_duct_segment, "ds_density"].iloc[0] * 
        filtered_duct_segment_catalog.loc[filtered_duct_segment_catalog["ds_id"] == selected_duct_segment, "ds_ef_pkg"].iloc[0] +
        duct_fitting_volume * duct_fitting_catalog.loc[duct_fitting_catalog["df_id"] == selected_duct_fitting, "df_density_pm3"].iloc[0] * 
        duct_fitting_catalog.loc[duct_fitting_catalog["df_id"] == selected_duct_fitting, "df_ef_pkg"].iloc[0]
    )

    total_cost = (
        diffuser_count * filtered_diffuser_catalog.loc[filtered_diffuser_catalog["ad_id"] == selected_diffuser, "ad_cost_punit"].iloc[0] +
        duct_segment_volume * filtered_duct_segment_catalog.loc[filtered_duct_segment_catalog["ds_id"] == selected_duct_segment, "ds_density"].iloc[0] * 
        filtered_duct_segment_catalog.loc[filtered_duct_segment_catalog["ds_id"] == selected_duct_segment, "ds_cost_pkg"].iloc[0] +
        duct_fitting_volume * duct_fitting_catalog.loc[duct_fitting_catalog["df_id"] == selected_duct_fitting, "df_density_pm3"].iloc[0] * 
        duct_fitting_catalog.loc[duct_fitting_catalog["df_id"] == selected_duct_fitting, "df_cost_pkg"].iloc[0]
    )

    return {
        "products": {
            "air_diffuser": selected_diffuser,
            "duct_segment": selected_duct_segment,
            "duct_fitting": selected_duct_fitting,
        },
        "total_carbon": total_carbon,
        "total_cost": total_cost,
    }

# 3. Run Conversation
def run_conversation():
     # Initial user message
    messages = [
        {"role": "system", "content": "You are a helpful assistant for ductwork design. Please use the suitbale tools to answer the user's question."},
        {"role": "user", "content": "You are given a json path named 'knowledge_graph_case_study.json', which includes a ductwork design dictionary. What is the total embodied carbon and cost of this ductwork design?"}]
        

    tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_stats",
                    "description": "Read the json file as a design dictionary and calculate total embodied carbon and cost of this ductwork design",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "design": {
                                "type": "string",
                                "description": "The path of a json file that includes a dictionary variable of the ductwork design"
                            }
                        },
                        "required": ["design"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_lowest_carbon_combination",
                    "description": "Find the optimal combination of air diffusers, duct segments, and duct fittings that minimizes carbon emissions while meeting design requirements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "json_file_path": {
                                "type": "string",
                                "description": "Path to the JSON file containing the current design"
                            },
                            "diffuser_catalog": {
                                "type": "string",
                                "description": "Path of the DataFrame containing air diffuser catalog data"
                            },
                            "duct_segment_catalog": {
                                "type": "string",
                                "description": "Path of the DataFrame containing duct segment catalog data"
                            },
                            "duct_fitting_catalog": {
                                "type": "string",
                                "description": "Path of the DataFrame containing duct fitting catalog data"
                            },
                            "json_size_mm": {
                                "type": "number",
                                "description": "Required size in millimeters for the air diffuser"
                            },
                            "min_airflow_cfm": {
                                "type": "number",
                                "description": "Minimum required airflow in CFM"
                            },
                            "max_noise_nc": {
                                "type": "number",
                                "description": "Maximum allowed noise level in NC"
                            },
                            "max_velocity_ms": {
                                "type": "number",
                                "description": "Maximum allowed velocity in meters per second"
                            }
                        },
                        "required": [
                            "json_file_path",
                            "diffuser_catalog",
                            "duct_segment_catalog",
                            "duct_fitting_catalog",
                            "json_size_mm",
                            "min_airflow_cfm",
                            "max_noise_nc",
                            "max_velocity_ms"
                        ]
                    }
                }
            }
        ]
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        # tool_choice="required",
    )

    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)

    print("Model's response:")  
    print(response_message)  

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f"Function call: {function_name}")  
            print(f"Function arguments: {function_args}")  
            
            if function_name == "get_current_stats":
                function_response = get_current_stats(
                    design=function_args.get("design")
                )
            elif function_name == "find_lowest_carbon_combination":
                function_response = find_lowest_carbon_combination(
                    json_file_path=function_args.get("json_file_path"),
                    diffuser_catalog=function_args.get("diffuser_catalog"),
                    duct_segment_catalog=function_args.get("duct_segment_catalog"),
                    duct_fitting_catalog=function_args.get("duct_fitting_catalog"),
                    json_size_mm=function_args.get("json_size_mm"),
                    min_airflow_cfm=function_args.get("min_airflow_cfm"),
                    max_noise_nc=function_args.get("max_noise_nc"),
                    max_velocity_ms=function_args.get("max_velocity_ms")
                )
            else:
                function_response = json.dumps({"error": "Unknown function"})
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
    else:
        print("No tool calls were made by the model.")  

    # Second API call: Get the final response from the model
    result_str = ''
    # Format the results in a more structured way for better analysis
    result_str += "Total Values:\n"
    result_str += f"Total Embodied Carbon: {function_response['total_carbon']} kgCO2e\n"
    result_str += f"Total Cost: {function_response['total_cost']} HKD\n\n"
    
    result_str += "Component-wise Breakdown:\n"
    result_str += "Air Terminal:\n"
    result_str += f"- Carbon: {function_response['air_terminal_carbon']} kgCO2e\n"
    result_str += f"- Cost: {function_response['air_terminal_cost']} HKD\n\n"
    
    result_str += "Duct Segment:\n"
    result_str += f"- Carbon: {function_response['duct_segment_carbon']} kgCO2e\n"
    result_str += f"- Cost: {function_response['duct_segment_cost']} HKD\n\n"
    
    result_str += "Duct Fitting:\n"
    result_str += f"- Carbon: {function_response['duct_fitting_carbon']} kgCO2e\n"
    result_str += f"- Cost: {function_response['duct_fitting_cost']} HKD\n"

    new_messages = [
        {"role": "system", "content": "You are a helpful assistant for ductwork design analysis. Analyze the carbon and cost hotspots in the ductwork design based on the component-wise breakdown provided."},
        {"role": "user", "content": f"Based on the following detailed breakdown of embodied carbon and cost values, perform a hotspot analysis of the ductwork design. Identify which components contribute most to the total carbon and cost, and suggest potential areas for optimization. Results:\n\n{result_str}"}
    ]
    final_response = client.chat.completions.create(
        model=deployment_name,
        messages=new_messages,
        temperature=0.2,
        max_tokens=1024
    )

    return final_response.choices[0].message.content

# Run the conversation and print the result
print(run_conversation())


