from http.server import HTTPServer, BaseHTTPRequestHandler

hostName = "192.168.1.12"
hostPort = 8000
class Serv(BaseHTTPRequestHandler):

    def do_GET(self):
        readData=0;
        if self.path == '/':
            self.path = '/index.html'
            path=self.path.split('?')[0]
        if self.path.split('?')[0]=='/read_data':
            readData=1;
        try:
            file_to_open = open(self.path[1:]).read()
            self.send_response(200)
        except:
            file_to_open = "File not found"
            self.send_response(404)
        self.end_headers()
        if readData==0:
            self.wfile.write(bytes(file_to_open, 'utf-8'))
        else:
            path=self.path.split('?')
            if (len(path)>1):
             data=self.path.split('?')[1]
            pars=data.split('&')
            res=""
            for par in pars:
                res+="name: "+par.split('=')[0]+" value: "+par.split('=')[1]+"\n"
            self.wfile.write(bytes(res, 'utf-8'))
    def do_POST(self):
        self.path = '/index.html'

httpd = HTTPServer((hostName, hostPort), Serv)
httpd.serve_forever()
