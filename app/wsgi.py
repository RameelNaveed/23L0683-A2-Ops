"""Entry point for the production server.

Docker:  gunicorn app.wsgi:app
Windows: waitress-serve --listen=127.0.0.1:5000 app.wsgi:app
Flask's own server (debug OFF): flask --app app.wsgi run
"""
from app import create_app

app = create_app()
