import sys


def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message including the file name
    and line number where the exception occurred.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        f"Error occurred in Python script: [{file_name}] "
        f"at line number: [{exc_tb.tb_lineno}] "
        f"Error message: [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom exception class for the project.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 10
        b = 0
        c = a / b

    except Exception as e:
        raise CustomException(e, sys)