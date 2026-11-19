from datetime import datetime 
import requests
from utilities.utils import log
import json

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url , timeout = 10)
    response.raise_for_status()
    return response.text


#TOOLS = [{TOOL1} , {TOOL2} , {TOOL3}]

TOOLS = [
    {
        "type":"function",
        "function":{
            "name" : "get_current_time",
            "description" : "Return current date and time.",
            "parameters": {
                "type":"object",
                "properties" : {}
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name": "get_city_current_weather",
            "description":"this tool retuns the current weather of a given city.",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"city to get the weather for."
                    }
                },
                "required":["city"]
            },
        },
    },
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time" : get_current_time,
    "get_city_current_weather":get_city_weather,
}


def run_tool(tool_call):
    func_name = tool_call.function.name
    func_args = json.loads(     #convert string to dictionary 
        tool_call.function.arguments or "{}"
       )
    log(f"{func_name}({func_args})")
    fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]

    result = fn(**func_args)     #convert disct to args like city = "aligarh" , country = India
    log(f"   ->{result}")
    return str(result)