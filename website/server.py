from gevent.pywsgi import WSGIServer
from app import application

http_server = WSGIServer(('0.0.0.0', 1229), application)
http_server.serve_forever()