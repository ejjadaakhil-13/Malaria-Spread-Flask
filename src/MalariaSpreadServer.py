import os
import re
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyodbc
import seaborn as sns
from flask import Flask, redirect, render_template, request, url_for
from sklearn.metrics.pairwise import cosine_similarity

from Constants import connString
from RoleModel import RoleModel
from UsersModel import UsersModel

app = Flask(__name__)
app.secret_key = "MySecret"
ctx = app.app_context()
ctx.push()

with ctx:
    pass
user_id = ""
emailid = ""
role_object = None
message = ""
msgType = ""
uploaded_file_name = ""


def initialize():
    global message, msgType
    message = ""
    msgType = ""


def process_role(option_id):
    if option_id == 4:
        if role_object.canRole == False:
            return False

    if option_id == 6:
        if role_object.canUsers == False:
            return False

    return True


@app.route("/")
def index():
    global user_id, emailid
    return render_template("Login.html")


@app.route("/processLogin", methods=["POST"])
def processLogin():
    global user_id, emailid, role_object
    emailid = request.form["emailid"]
    password = request.form["password"]

    if emailid != 'malaria' and password != '1':
        return render_template("Login.html", processResult="Invalid Role")

    role_object = RoleModel(1, 1, 1, 1, 1, 1, 1, 1, 1)

    return render_template("Dashboard.html")


@app.route("/ChangePassword")
def changePassword():
    global user_id, emailid
    return render_template("ChangePassword.html")


@app.route("/ProcessChangePassword", methods=["POST"])
def processChangePassword():
    global user_id, emailid
    oldPassword = request.form["oldPassword"]
    newPassword = request.form["newPassword"]
    confirmPassword = request.form["confirmPassword"]
    conn1 = pyodbc.connect(connString, autocommit=True)
    cur1 = conn1.cursor()
    sqlcmd1 = "SELECT * FROM Users WHERE emailid = '" + emailid + "' AND password = '" + oldPassword + "'";
    cur1.execute(sqlcmd1)
    row = cur1.fetchone()
    cur1.commit()
    if not row:
        return render_template("ChangePassword.html", msg="Invalid Old Password")

    if newPassword.strip() != confirmPassword.strip():
        return render_template("ChangePassword.html", msg="New Password and Confirm Password are NOT same")

    conn2 = pyodbc.connect(connString, autocommit=True)
    cur2 = conn2.cursor()
    sqlcmd2 = "UPDATE Users SET password = '" + newPassword + "' WHERE emailid = '" + emailid + "'";
    cur1.execute(sqlcmd2)
    cur2.commit()
    return render_template("ChangePassword.html", msg="Password Changed Successfully")


@app.route("/Dashboard")
def Dashboard():
    global user_id, emailid
    return render_template("Dashboard.html")


@app.route("/Information")
def Information():
    global message, msgType
    return render_template("Information.html", msgType=msgType, message=message)


def get_datasets():
    file_list = os.listdir("static/Dataset")
    print(file_list)
    return file_list


df = None


def load_df():
    global df
    if df is None:
        cases_df = pd.read_csv(f"static/Dataset/cases.csv")
        death_df = pd.read_csv(f"static/Dataset/deaths.csv")
        country_list = list(cases_df['Country'])
        year_list = list(cases_df.drop('Country', axis=1).columns)
        df = pd.DataFrame()
        idx = 0
        for coun in country_list:
            for yr in year_list:
                try:
                    df.loc[idx, 'Country'] = coun
                    df.loc[idx, "Year"] = yr
                    df.loc[idx, "Confirmed"] = cases_df[(cases_df['Country'] == coun)][yr].values[0]
                    df.loc[idx, "Deaths"] = death_df[(death_df['Country'] == coun)][yr].values[0]
                    idx += 1
                except:
                    pass
        df.fillna(0, inplace=True)


load_df()


@app.route("/DatasetListing")
def dataset_listing_operation():
    file_list = get_datasets()
    return render_template("DatasetListing.html", fileList=file_list)


@app.route("/DatasetView")
def dataset_view():
    dataset_name = request.args.get("datasetName")
    records = pd.read_csv(f"static/Dataset/{dataset_name}")
    return render_template("DatasetView.html", records=records)


def get_categories():
    category_set = set()
    for key, value in df["category"].items():
        category_split = value.split("|")
        category_set.add(category_split[len(category_split) - 1])
    return sorted(list(category_set))


@app.route("/TotalCasesTreeView")
def total_cases_tree_view():
    return render_template("TotalCasesTreeView.html", imageFileName="")

def generate_total_cases_treemap():
    color_scale = px.colors.qualitative.Plotly

    columns = ['Confirmed', 'Deaths']

    # Create a dictionary to specify labels for columns
    labels = {
        'Confirmed': 'Total Confirmed Cases',
        'Deaths': 'Total Deaths'
    }

    for col in columns:
        # Create the treemap
        tc = df.groupby(by="Country", as_index=False).sum()
        tc = tc[tc[col] > 10]
        print(tc)
        fig = px.treemap(
            tc,
            values=col,
            path=['Country'],
            color=col,
            color_continuous_scale=color_scale
        )

        # Customize the layout
        fig.update_layout(
            title=f'Treemap of {labels[col]} by Country',
            margin=dict(l=0, r=0, b=0, t=40),
        )

        # Customize the color scale and labels
        fig.update_traces(
            marker_line_width=1.5,
            hovertemplate='<b>%{label}</b><br>%{value}<extra></extra>',
            textinfo='label+value+percent parent'
        )

        fig.write_image(f"static/charts/Treemap-{col}.png")

@app.route("/ProcessTotalCasesTreeView", methods=['POST'])
def process_total_cases_tree_view():
    generate_total_cases_treemap()
    return render_template("TotalCasesTreeView.html", imageFileName="AAA")


def generate_top_pie(field, ntop, category):
    columns = [category]

    for col in columns:
        tc = df.groupby(by=field, as_index=False)[col].sum().sort_values(by=col, ascending=False)
        fig = px.pie(tc.head(int(ntop)), names=field, values=col,
                     title=f'Top {ntop} {field} by {col} Cases',
                     template='plotly_dark',
                     )

        fig.write_image(f"static/charts/TopPie.png")

@app.route("/TopPie")
def top_pie_view():
    return render_template("TopPieView.html", imageFileName="")

@app.route("/ProcessTopPie", methods=['POST'])
def process_top_pie_view():
    field = request.form["field"]
    ntop = request.form["ntop"]
    category = request.form["category"]
    generate_top_pie(field, ntop, category)
    return render_template("TopPieView.html", field=field, ntop=ntop, category=category, imageFileName="AAA")


def generate_top_area(field, ntop, category):
    columns = [category]

    for col in columns:
        tc = df.groupby(by=field, as_index=False)[col].sum().sort_values(by=col, ascending=False)
        fig = px.area(tc.head(int(ntop)), x=field, y=col, color=field, template='plotly_dark',
                      title=f'{field} Top {ntop} {category} Cases over time: Area Plot', color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.write_image(f"static/charts/TopArea.png")

@app.route("/TopArea")
def top_area_view():
    return render_template("TopAreaView.html", imageFileName="")

@app.route("/ProcessTopArea", methods=['POST'])
def process_top_area_view():
    field = request.form["field"]
    ntop = request.form["ntop"]
    category = request.form["category"]
    generate_top_area(field, ntop, category)
    return render_template("TopAreaView.html", field=field, ntop=ntop, category=category, imageFileName="AAA")


def generate_top_bar(field, ntop, category):
    columns = [category]

    for col in columns:
        tc = df.groupby(by=field, as_index=False)[col].sum().sort_values(by=col, ascending=False)
        fig = px.bar(tc, x=field, y=col, color=field,
                     title=f'Bar Comparison for top {ntop} {field} on {col}',
                     template='plotly_dark', color_discrete_sequence=px.colors.sequential.Plasma_r
                     )
        fig.write_image(f"static/charts/TopBar.png")

@app.route("/TopBar")
def top_bar_view():
    return render_template("TopBarView.html", imageFileName="")

@app.route("/ProcessTopBar", methods=['POST'])
def process_top_bar_view():
    field = request.form["field"]
    ntop = request.form["ntop"]
    category = request.form["category"]
    generate_top_bar(field, ntop, category)
    return render_template("TopBarView.html", field=field, ntop=ntop, category=category, imageFileName="AAA")


def generate_correlation_matrix():
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Columns')
    plt.savefig(f"static/charts/CorrelationMatrix.png")

@app.route("/CorrelationMatrix")
def correlation_matrix_view():
    return render_template("CorrelationMatrixView.html", imageFileName="")

@app.route("/ProcessCorrelationMatrixView", methods=['POST'])
def process_correlation_matrix_view():
    generate_correlation_matrix()
    return render_template("CorrelationMatrixView.html",  imageFileName="AAA")


def generate_similar_unrelated_countries(country):
    group = df.groupby("Country")
    vectors = []
    countries = sorted(df['Country'].unique())

    def find_similar_countries(country, k):
        i = countries.index(country)
        return [countries[j] for j in sim[i].argsort()[::-1][1:k + 1]]

    def find_opposite_countries(country, k):
        i = countries.index(country)
        return [countries[j] for j in sim[i].argsort()[1:k + 1]]

    def get_results(country, k):
        similar = find_similar_countries(country, k=k)
        opposite = find_opposite_countries(country, k=k)
        print(similar, type(similar))
        space = "\n\t - "

        print(f"Top {k} Countries that had cases similar to {country}:")
        print(space[1:] + space.join(similar))
        print(f"Top {k} Countries that had cases different to {country}:")
        print(space[1:] + space.join(opposite))
        return similar, opposite

    for country in countries:
        sub_group = group.get_group(country).drop(["Country", "Year"], axis=1)
        sub_group = sub_group.values
        sub_group = sub_group.ravel()
        vectors.append(sub_group)
    vectors = np.vstack(vectors)
    sim = cosine_similarity(vectors, vectors)
    similar, opposite = get_results("India", 10)
    return similar, opposite

@app.route("/SimilarityUnrelated")
def similarity_unrelated_view():
    countries = sorted(df['Country'].unique())
    return render_template("SimilarityUnrelatedView.html", countries=countries, imageFileName="")

@app.route("/ProcessSimilarityUnrelatedView", methods=['POST'])
def process_similarity_unrelated_view():
    country = request.form["country"]
    similar, opposite = generate_similar_unrelated_countries(country)
    print(similar, opposite)
    countries = sorted(df['Country'].unique())
    return render_template("SimilarityUnrelatedView.html",countries=countries, country=country, similar=similar, opposite=opposite,  imageFileName="AAA")


if __name__ == "__main__":
    app.run()

if __name__ == "__main__1":
    load_df()
    generate_total_cases_treemap()
    print(df, "XXXXXXXXXXXXXXXXX")
