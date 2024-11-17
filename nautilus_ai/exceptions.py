class NautilusAIException(Exception):
    """
    NautilusAI base exception. Handled at the outermost level.
    All other exception types are subclasses of this exception type.
    """


class OperationalException(NautilusAIException):
    """
    Requires manual intervention and will stop the node.
    Most of the time, this is caused by an invalid Configuration.
    """
