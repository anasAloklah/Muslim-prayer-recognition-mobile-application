from http.server import HTTPServer, BaseHTTPRequestHandler
from prediction import  predcition
hostName = "192.168.1.15"
hostPort = 8000
class Serv(BaseHTTPRequestHandler):


    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if self.path=='/check':
            self.wfile.write(bytes("test", 'utf-8'))

    def do_POST(self):
        self.send_response(200)
        self.end_headers()
        content_length=int(self.headers['Content-Length'])
        #print('content_length ',content_length)
        post_data = self.rfile.read(content_length)
        post_data=post_data.decode('utf-8')
        pars = post_data.split('&')
        dataSensor=""
        type=""
        for par in pars:
            if (par.split('=')[0] == 'data'):
                dataSensor = par.split('=')[1]
            if (par.split('=')[0] == 'Type'):
                type = par.split('=')[1]

        dataSensor=dataSensor.replace("%3B",";")
        dataSensor = dataSensor.replace("%7C", "|")
        #print(dataSensor)
        if self.path == '/read_data':
            res=predcition(dataSensor, type)
            print(res)
            self.wfile.write(bytes(str(res),'utf-8'))

httpd = HTTPServer((hostName, hostPort), Serv)
httpd.serve_forever()
