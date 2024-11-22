import uuid
import datetime
import base64
from pyDes import *
class LicenseGenerate():
    def __init__(self):
        self.pw = 'synthere2024zzz'
        self.Des_Key = "ZYNTHERE"  # Key b"DESCRYPT"#
        self.Des_IV = "ZYNTHERE"  # iv vect b"\x15\1\x2a\3\1\x23\2\0"#
    def get_str_for_license(self, valid_days):
        mac_addr = self.get_mac_address()
        print("Received validays: {}, mac_addr: {}".format(valid_days, mac_addr))
        psw = self.pw#self.hash_msg(self.pw)
        current_time = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
        end_date = (datetime.datetime.now() + datetime.timedelta(days=valid_days)).strftime("%Y-%m-%d %H:%M:%S")

        license_str = {}
        #license_str['time_start'] = current_time
        license_str['time_end'] = end_date
        license_str['psw'] = psw
        s = str(license_str)
        return s

    def get_mac_address(self):
        mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
        return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])

    def hash_msg(self, msg):
        import hashlib
        sha256 = hashlib.sha256()
        sha256.update(msg.encode('utf-8'))
        res = sha256.hexdigest()
        return res

    def Encrypted(self, tr):
        print("len tr:", len(tr))
        k = des(self.Des_Key, CBC, self.Des_IV, pad=None, padmode=PAD_PKCS5)
        EncryptStr = k.encrypt(tr)
        print("encr len:", len(EncryptStr), len(base64.b64encode(EncryptStr)))
        # EncryptStr = binascii.unhexlify(k.encrypt(str))
        ###  print('regis：',base64.b64encode(EncryptStr))
        return base64.b64encode(EncryptStr) ##bytes

    #1 generate by valid time and psw
    def generate_lic(self):
        valid_duration = 7
        s = self.get_str_for_license(valid_duration)
        lic = self.Encrypted(s)
        return lic

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

    return

if __name__ == "__main__":
    main()