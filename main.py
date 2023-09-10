# AI-Powered Personal Finance Assistant

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class PersonalFinanceAssistant:
    def __init__(self, username):
        self.username = username
        self.income = []
        self.expenses = []
        self.categories = []
        self.goals = []
        self.portfolio = []
        self.expense_predictions = []

    def track_income(self, amount):
        self.income.append(amount)
        print("Income tracked successfully!")

    def track_expense(self, amount, category):
        self.expenses.append(amount)
        self.categories.append(category)
        print("Expense tracked successfully!")

    def set_goal(self, goal):
        self.goals.append(goal)
        print("Goal set successfully!")

    def add_to_portfolio(self, investment):
        self.portfolio.append(investment)
        print("Investment added successfully!")

    def analyze_expenses(self):
        df = pd.DataFrame(
            {'Expense': self.expenses, 'Category': self.categories})
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Category', y='Expense', data=df)
        plt.xticks(rotation=45)
        plt.xlabel('Category')
        plt.ylabel('Expense')
        plt.title('Expense Breakdown')
        plt.show()

    def generate_budget(self):
        df = pd.DataFrame(
            {'Expense': self.expenses, 'Category': self.categories})
        df['Category'] = df['Category'].astype('category').cat.codes
        X = df[['Category']]
        y = df['Expense']
        model = LinearRegression()
        model.fit(X, y)
        budget = model.predict(X)
        df['Budget'] = budget
        df['Budget'] = np.where(df['Budget'] < 0, 0, df['Budget'])
        df['Budget'] = np.round(df['Budget'], 2)
        df['Difference'] = df['Budget'] - df['Expense']
        df['Difference'] = np.where(df['Difference'] < 0, 0, df['Difference'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Category', y='Budget', data=df)
        plt.xticks(rotation=45)
        plt.xlabel('Category')
        plt.ylabel('Budget')
        plt.title('Budget Breakdown')
        plt.show()

    def track_expense_predictions(self):
        if len(self.expenses) < 10:
            print("Insufficient data to generate expense predictions.")
            return

        df = pd.DataFrame(
            {'Expense': self.expenses, 'Category': self.categories})
        kmeans = KMeans(n_clusters=2)
        X = df[['Category', 'Expense']]
        X_scaled = StandardScaler().fit_transform(X)
        kmeans.fit(X_scaled)
        df['Cluster'] = kmeans.labels_
        df['Category'] = df['Category'].astype('category').cat.codes

        predict_df = pd.DataFrame(
            {'Category': df['Category'], 'Cluster': df['Cluster']})
        predict_df = pd.get_dummies(
            predict_df, columns=['Category', 'Cluster'], prefix='', prefix_sep='')
        missing_categories = set(range(10)) - set(predict_df.columns)
        for category in missing_categories:
            predict_df[category] = 0

        model = LinearRegression()
        model.fit(df[['Category', 'Cluster']], df['Expense'])
        self.expense_predictions = model.predict(predict_df)
        self.expense_predictions = np.round(self.expense_predictions, 2)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.expense_predictions)+1),
                 self.expense_predictions, 'o-', label='Predicted Expense')
        plt.xlabel('Expense')
        plt.ylabel('Amount')
        plt.title('Expense Predictions')
        plt.legend()
        plt.show()

    def visualize_portfolio(self):
        df = pd.DataFrame({'Investment': self.portfolio})
        df['Investment'] = df['Investment'].astype('float')
        df['Investment'].plot.pie(autopct='%.2f%%', figsize=(6, 6))
        plt.axis('equal')
        plt.title('Portfolio Allocation')
        plt.show()


# Demo
assistant = PersonalFinanceAssistant("John")
assistant.track_income(5000)
assistant.track_expense(1000, "Rent")
assistant.track_expense(500, "Groceries")
assistant.track_expense(50, "Transportation")
assistant.track_expense(200, "Entertainment")
assistant.analyze_expenses()
assistant.generate_budget()
assistant.set_goal("Save $1000 for vacation")
assistant.add_to_portfolio("Stocks")
assistant.add_to_portfolio("Bonds")
assistant.visualize_portfolio()
assistant.track_expense_predictions()
