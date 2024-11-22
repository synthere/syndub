# -*- coding: utf-8 -*-
import os
import time
import requests
import config
import tools


shound_del=False
def update_proxy(type='set'):
    global shound_del
    if type=='del' and shound_del:
        del os.environ['http_proxy']
        del os.environ['https_proxy']
        del os.environ['all_proxy']
        shound_del=False
    elif type=='set':
        raw_proxy=os.environ.get('http_proxy')
        if not raw_proxy:
            proxy=tools.set_proxy()
            if proxy:
                shound_del=True
                os.environ['http_proxy'] = proxy
                os.environ['https_proxy'] = proxy
                os.environ['all_proxy'] = proxy

def trans(text_list, target_language="en", *, set_p=True,inst=None,stop=0,source_code=""):
    """
    text_list:
    target_language:
        
    set_p:
    """

    target_text = []
    index = -1  
    iter_num = 0 
    err = ""
    proxies=None
    pro=update_proxy(type='set')
    if pro:
        proxies={"https":pro,"http":pro}

    headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
    def get_content(data,auth):
        url = f"https://api-edge.cognitive.microsofttranslator.com/translate?from=&to={data['target_language']}&api-version=3.0&includeSentenceLength=true"
        headers['Authorization'] = f"Bearer {auth.text}"
        config.logger.info(f'[Mircosoft]请求数据:{url=},{auth.text=}')
        response = requests.post(url, json=[{"Text": data['text']}], headers=headers, timeout=300)
        config.logger.info(f'[Mircosoft]返回:{response.text=}')
        if response.status_code != 200:
            raise Exception(f'{response.status_code=}')
        try:
            re_result = response.json()
        except Exception:
            raise Exception('notjson' + response.text)

        if len(re_result) == 0 or len(re_result[0]['translations']) == 0:
            raise Exception(f'{re_result}')
        return re_result[0]['translations'][0]['text']
    print("to translate")
    while 1:
        iter_num += 1
        if iter_num > 3:
            print("max time reached")
            break
        if iter_num > 1:
            if set_p:
                tools.set_process(
                    f"第{iter_num}次出错重试" if config.defaulelang == 'zh' else f'{iter_num} retries after error',btnkey=inst.init['btnkey'] if inst else "")
            time.sleep(5)
        #print("to zli")
        if isinstance(text_list, str):
            source_text = text_list.strip().split("\n")
        else:
            source_text = [f"{t['text']}" for t in text_list]
        
        #print("to split")
        
        split_size = int(config.settings['trans_thread'])
        split_source_text = [source_text[i:i + split_size] for i in range(0, len(source_text), split_size)]
        try:
            auth=requests.get('https://edge.microsoft.com/translate/auth',headers=headers,proxies=proxies)
        except:
            err='连接微软翻译失败，请更换其他翻译渠道' if config.defaulelang=='zh' else 'Failed to connect to Microsoft Translate, please change to another translation channel'
            continue
        #print(" to proceds:", split_source_text)
        for i,it in enumerate(split_source_text):
            print("i:", i, "it:", it)

            if i <= index:
                print("i", i, "ind:", index)
                continue
            if stop>0:
                print("to sleep stop:", stop)
                time.sleep(stop)
            print("to get")
            try:
                source_length=len(it)
                text = "\n".join(it)
                data={
                    "text":text,
                    "target_language":target_language
                }
                res_trans=get_content(data,auth)
                print("trans return result:", res_trans)
                result=tools.cleartext(res_trans).split("\n")
                result_length = len(result)
                # not match split again
                if result_length < source_length:
                    print(f'翻译前后数量不一致，需要重新切割')
                    result=[]
                    for line_res in it:
                        data['text']=line_res
                        result.append(get_content(data,auth))

                if inst and inst.precent < 75:
                    inst.precent += round((i + 1) * 5 / len(split_source_text), 2)

                result_length=len(result)

                while result_length<source_length:
                    result.append("")
                    result_length+=1
                result=result[:source_length]
                target_text.extend(result)
            except Exception as e:
                err=f'{str(e)}'
                break
            else:
                err=''
                index= i
                iter_num=0
        else:
            break

    update_proxy(type='del')

    if err:
        config.logger.error(f'[Mircosoft]翻译请求失败:{err=}')
        raise Exception(f'Mircosoft:{err}')

    if isinstance(text_list, str):
        return "\n".join(target_text)

    max_i = len(target_text)
    if max_i < len(text_list)/2:
        raise Exception(f'Mircosoft:{"translate errr"}')

    for i, it in enumerate(text_list):
        if i < max_i:
            text_list[i]['text'] = target_text[i]
        else:
            text_list[i]['text'] = ""
    return text_list
