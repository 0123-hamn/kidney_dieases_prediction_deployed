import sys


def error_message_detail(error_message, error_details: sys):
    _, _, exc_tb = error_details.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return f"""
Error occurred in python script:
File name: {file_name}
Line number: {line_number}
Error message: {str(error_message)}
"""


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self):
        return self.error_message
