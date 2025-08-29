from utils import parser_path,folder_split,function_in_class

def depd_context(project_name,completion_path,dependencys):
    context_dict={}
    for depend in dependencys:
        dependency_path,dependency_class,dependency_element=parser_path(project_name,completion_path,depend) 
        if(dependency_path==None):
            continue
        codes=folder_split(dependency_path) 
        if(dependency_element!=None):
            if(dependency_class!=None and dependency_class in codes.keys()):
                code=codes[dependency_class][1]
                if(codes[dependency_class][0]=="function"):
                    if(dependency_path not in context_dict.keys()):
                        import_and_expression=codes["import_and_expression"]
                        dependency_element=codes[dependency_class][1]+"\n"
                        context_dict[dependency_path]=[import_and_expression,dependency_element]
                    else:
                        context_dict[dependency_path][1]+=codes[dependency_class][1]+"\n"
                    continue
                ##依赖在class内
                funs=function_in_class(code)
                if(dependency_path+"_"+dependency_class not in context_dict.keys()):
                    class_statement=codes[dependency_class][2]+"\n"
                    import_and_expression=funs["import_and_expression"]
                    if("__init__" in funs.keys()):
                        init=funs["__init__"][1]+"\n"
                    else:
                        init=""
                    if(dependency_element in funs.keys() and dependency_element!="__init__"):
                        dependency_element=codes[dependency_class][1]+"\n"
                    else:
                        dependency_element=""
                    context_dict[dependency_path+"_"+dependency_class]=[class_statement,import_and_expression,init]
                else:
                    if(dependency_element in funs.keys() and dependency_element!="__init__"):
                        dependency_element=codes[dependency_class][1]+"\n"
                    else:
                        dependency_element=""
                    context_dict[dependency_path+"_"+dependency_class][2]+=dependency_element
    
            else:
                if(dependency_element not in codes.keys()):
                    continue
                if(dependency_path not in context_dict.keys()):
                    import_and_expression=codes["import_and_expression"]
                    dependency_element=codes[dependency_element][1]+"\n"
                    context_dict[dependency_path]=[import_and_expression,dependency_element]
                else:
                    context_dict[dependency_path][1]+=codes[dependency_element][1]+"\n"
    context=""
    for v in context_dict.values():
       for item in v:
           context+=item
    return context



def Construct_context(project_name,completion_path,all_dependencys):
    All_context=""

    context=depd_context(project_name,completion_path,all_dependencys['intra_class'])
    if(context !=""):
        All_context+="#The context on the Same Class:\n"+context+"\n"

    context=depd_context(project_name,completion_path,all_dependencys['intra_file'])
    if(context !=""):
        All_context+="#The context on the Same File:\n"+context+"\n"
    
    context=depd_context(project_name,completion_path,all_dependencys['cross_file'])
    if(context !=""):
        All_context+="#The context in the Other File:\n"+context+"\n"
    
    return All_context