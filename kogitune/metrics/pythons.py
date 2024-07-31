import ast
import traceback

def get_syntax_error_line(code):
    try:
        ast.parse(code)
        return None  # エラーがない場合はNoneを返す
    except SyntaxError as e:
        return e.lineno  # エラーが発生した行番号を返す

def clean_code(code):
    while True:
        error_lineno = get_syntax_error_line(code)
        if error_lineno is None:
            return code
        if '\n' not in code:
            break
        code, _, _ = code.rpartition('\n')
    return None

def extract_python_code(text):
    result = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].strip() == '':
            # 空行はスキップする
            i += 1
            continue
        code = '\n'.join(lines[i:])
        next = get_syntax_error_line(code)
        #print(i, next, code)
        if next == 1:
            # 先頭でエラーが発生したらスキップする
            i += 1
            continue
        if next is None:
            result.append(code)
            break
        code = clean_code('\n'.join(lines[i:i+next-1]))
        if code is not None:
            result.append(code)
        i += next
    return '\n'.join(result)

def extract_from_code_completion(prompt, generated_text):
    stop_sequences=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
    min_stop_index = len(generated_text)
    for seq in stop_sequences:
        stop_index = generated_text.find(seq)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    code = prompt + "\n" + generated_text[:min_stop_index]
    return extract_python_code(code)

TEMPLATE_CODE_FIX = '''\
The following error has occurred. 
Please fix the code so that it can be executed without errors.

### Error
{error_message}
{stack_trace}

### Code
{code}

'''

def get_code_fix_prompt(code_str, test_code):
    if isinstance(code_str, list):
        return [get_code_fix_prompt(x, test_code) for x in code_str]
    try:
        # コードを実行
        exec(code_str+test_code)
        return None
    except Exception as e:
        # エラーが発生した場合、エラーメッセージとスタックトレースを回収
        error_message = str(e)
        stack_trace = traceback.format_exc()
        prompt = TEMPLATE_CODE_FIX.format(
            error_message=error_message, 
            stack_trace=stack_trace[:256], 
            code=code_str+test_code)
        return prompt

