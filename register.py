# -*- coding: utf-8 -*-

import uuid
import datetime

import base64
from pyDes import *
#from cryptography.fernet import Fernet
"""
    def generate_aes_key(self):
        key = Fernet.generate_key() #SYNTHEREZ
        return key
    def aes_en(self, msg):
        cipher_suite = Fernet(self.key)
        return cipher_suite.encrypt(msg)
    def aes_de(self, msg):
        cipher_suite = Fernet(self.key)
        return cipher_suite.decrypt(msg)
"""
class LicRegister:
    def __init__(self):
        self.key = ''
        self.Des_Key = "ZYNTHERE"  # Key b"DESCRYPT" #
        self.Des_IV = "ZYNTHERE"  # iv vect b"\x15\1\x2a\3\1\x23\2\0"#
        self.pw = 'synthere2024zzz'
    def get_mac_address(self):
        mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
        return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])
    def hash_msg(self, msg):
        import hashlib
        sha256 = hashlib.sha256()
        sha256.update(msg.encode('utf-8'))
        res = sha256.hexdigest()
        return res

    # DES+base64 en
    def Encrypted(self, tr):
        k = des(self.Des_Key, CBC, self.Des_IV, pad=None, padmode=PAD_PKCS5)
        EncryptStr = k.encrypt(tr)
        # EncryptStr = binascii.unhexlify(k.encrypt(str))
        ###  print('regis：',base64.b64encode(EncryptStr))
        return base64.b64encode(EncryptStr)

    # #des+base64解码
    def DesDecrypt(self,tr):
        btr = base64.b64decode(tr)
        k = des(self.Des_Key, CBC, self.Des_IV, pad=None, padmode=PAD_PKCS5)
        DecryptStr = k.decrypt(btr)
        return DecryptStr

    def get_str_for_license(self, valid_days):
        print("Received validays: {}".format(valid_days))
        psw = self.pw#self.hash_msg('synthere2014zzz')
        current_time = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
        end_date = (datetime.datetime.now() + datetime.timedelta(days=valid_days)).strftime("%Y-%m-%d %H:%M:%S")

        license_str = {}
        license_str['time_start'] = current_time
        license_str['time_end'] = end_date
        license_str['psw'] = psw
        s = str(license_str)
        return s

    def check_license_date(self, lic_date):
        current_time = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
        current_time_array = datetime.datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        lic_date_array = datetime.datetime.strptime(lic_date, "%Y-%m-%d %H:%M:%S")
        remain_days = lic_date_array - current_time_array
        remain_days = remain_days.days
        if remain_days < 0 or remain_days == 0:
            return False
        else:
            return True

    def check_license_psw(self, psw):
        hashed_msg = self.pw#self.hash_msg(self.pw)
        if psw == hashed_msg:
            return True
        else:
            return False

    def generate_register_file_content(self):
        psw = self.pw#self.hash_msg(self.pw)
        current_time = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
        mac_addr = self.get_mac_address()

        license_str = {}
        license_str['mac'] = mac_addr
        license_str['time_start'] = current_time
        license_str['psw'] = psw
        s = str(license_str)
        return s

    #2 read licence and check validate time and psw
    def register(self, lic_num, reg_file):
        key = lic_num# this licence num
        # validate mac and datetime
        if key:
            key_decrypted = bytes(key, encoding='utf-8')
            lic_cont = self.DesDecrypt(key_decrypted)
            lic_dic = eval(lic_cont)
            print("lic", lic_dic)
            date_bool = self.check_license_date(lic_dic['time_end'])
            psw_bool = self.check_license_psw(lic_dic['psw'])
            if date_bool and psw_bool:
                rf = self.generate_register_file_content()
                rc = self.Encrypted(rf)
                print("register sucess")
                with open(reg_file, 'w') as f:
                    f.write(rc.decode('utf-8'))
                return True
            else:
                return False
        return False

    # #### open file and check mac
    def checkAuthored(self, reg_file):
        #check
        try:
            f = open(reg_file, 'r')
            if f:
                key = f.read()
                print('reg num is ：', key)
                if key:
                    lic_cont = self.DesDecrypt(key.encode('utf-8'))
                    lic_dic = eval(lic_cont)
                    mac_addr = self.get_mac_address()
                    print("mac add:", mac_addr)
                    if mac_addr == lic_dic['mac']:
                        print("valid lic file")
                        return True
            return False
        except:
            return False


def main():
    import licencegen
    lg = licencegen.LicenseGenerate()
    lic = lg.generate_lic()
    print("generate:", lic)
    with open("lic.bin", 'w') as fp:
        fp.write(lic.decode('utf-8'))
    with open('lic.bin', 'r') as fp:
        licr = fp.read()
    print("lcr", licr)
    lr = LicRegister()
    reg_file = 'reg.txt'
    lr.register(licr, reg_file)

    auth = lr.checkAuthored(reg_file)
    print("auth:", auth)
    return

if __name__ == "__main__":
    main()

