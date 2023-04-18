import xraydb as xdb
import pandas as pd
import plotly.express as px


def parameters(x):

    def actual_wrapper(func):

        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs, x=x)

        return wrapper
    return actual_wrapper


class Some:
    def __init__(self):
        self.y = 20

    @parameters(x=10)
    def summarize(self):
        nonlocal x
        return self.y + x


print(Some().summarize())