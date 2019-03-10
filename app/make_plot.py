import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)


def return_graphs():
    """Creates plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the plotly visualizations

    """

    # first chart
    graph_one = []

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_one.append(
        go.Bar(
            x=genre_names,
            y=genre_counts,
        )
    )

    layout_one = dict(title='Distribution of Message Genres',
                      xaxis=dict(title='Genre',),
                      yaxis=dict(title='Count'),
                      )

    # append all charts to the figures list
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))

    return graphs
