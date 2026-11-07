"""Code shared by training, the Flask app and the cloud model API.

Keeping preprocessing in one place is what guarantees that an image is
prepared the same way at training time and at serving time.
"""
