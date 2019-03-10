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
                      xaxis=dict(title='Genre',
                                 tickfont=dict(
                                     size=12,
                                     color='black'
                                 )),
                      yaxis=dict(title='Count',
                                 tickfont=dict(
                                     size=12,
                                     color='black'
                                 )),
                      )

    # second chart
    graph_two = []

    data_cateroy = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_name = [category.replace('_', ' ')
                     for category in data_cateroy.index.tolist()]
    category_count = data_cateroy.tolist()

    graph_two.append(
        go.Bar(
            x=category_name,
            y=category_count,
        )
    )

    layout_two = dict(title='Distribution of Messages Categories',
                      xaxis=dict(title='',
                                 tickangle=90,
                                 tickfont=dict(
                                     size=12,
                                     color='black'
                                 ),),
                      yaxis=dict(title='Count'),
                      tickfont=dict(
                          size=12,
                          color='black'
                      ))

    # append all charts to the figures list
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))

    return graphs
