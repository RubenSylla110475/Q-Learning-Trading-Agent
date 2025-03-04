import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


bac_data = pd.read_csv("bank_of_america.csv", usecols=["Date", "Close"])
bac_data.sort_values(by="Date", inplace=True) 
bac_data.reset_index(drop=True, inplace=True)

prices_bac = bac_data["Close"].values
T_bac = len(prices_bac)  # 1287 jours -> 1257

INITIAL_CASH = 5000
SHARE_BATCH = 10
MAX_SHARES = 100 

def get_portfolio_value(day, shares, cash, prices):
    return cash + shares * prices[day]

alpha = 0.1
gamma = 0.95
epsilon = 0.1 
num_episodes = 10000 

Q = {}
A = [0, 1, 2] 

def get_Q_state(day, shares, cash):
    cash_discret = int(round(cash, -2))
    state_key = (day, shares, cash_discret)
    if state_key not in Q:
        Q[state_key] = [0.0, 0.0, 0.0]
    return state_key

def epsilon_greedy_action(state_key):
    if random.random() < epsilon:
        return random.choice(A)
    else:
        return np.argmax(Q[state_key])


for episode in range(num_episodes):
    day = 0
    shares = 0
    cash = INITIAL_CASH

    while day < T_bac - 1:

        s = get_Q_state(day, shares, cash)

        a = epsilon_greedy_action(s)

        portfolio_before = get_portfolio_value(day, shares, cash, prices_bac)
        current_price = prices_bac[day]

        if a == 1:  # a = 1 : Buy
            cost = SHARE_BATCH * current_price
            if cash >= cost and shares + SHARE_BATCH <= MAX_SHARES:
                shares += SHARE_BATCH
                cash -= cost

        elif a == 2:  # a = 2 : Sell
            if shares >= SHARE_BATCH:
                shares -= SHARE_BATCH
                cash += SHARE_BATCH * current_price

        next_day = day + 1
        portfolio_after = get_portfolio_value(next_day, shares, cash, prices_bac)
        R = portfolio_after - portfolio_before  # récompense : variation de la valeur du portefeuille

        # s' : état futur
        s_prime = get_Q_state(next_day, shares, cash)
        best_next_action_value = max(Q[s_prime])
        current_q_value = Q[s][a]
        Q[s][a] = current_q_value + alpha * (R + gamma * best_next_action_value - current_q_value)

        day = next_day

    if (episode + 1) % 10 == 0:
        final_portfolio_value = get_portfolio_value(day, shares, cash, prices_bac)
        print(f"Episode {episode+1}/{num_episodes} - Portefeuille final BAC: {final_portfolio_value:.2f}")

day = 0
shares = 0
cash = INITIAL_CASH
while day < T_bac - 1:
    s = get_Q_state(day, shares, cash)
    a = np.argmax(Q[s])  # Optimal selection in A
    current_price = prices_bac[day]
    if a == 1:  # Buy
        cost = SHARE_BATCH * current_price
        if cash >= cost and shares + SHARE_BATCH <= MAX_SHARES:
            shares += SHARE_BATCH
            cash -= cost
    elif a == 2:  # Sell
        if shares >= SHARE_BATCH:
            shares -= SHARE_BATCH
            cash += SHARE_BATCH * current_price
    day += 1

final_value_bac = get_portfolio_value(day, shares, cash, prices_bac)
print(f"\n[POLITIQUE GREEDY] Valeur finale du portefeuille sur BAC : {final_value_bac:.2f}\n")

#---------------------GE------------------

ge_data = pd.read_csv("ge.csv", usecols=["Date", "Close"])
ge_data.sort_values(by="Date", inplace=True)
ge_data.reset_index(drop=True, inplace=True)

prices_ge = ge_data["Close"].values
T_ge = len(prices_ge)  # 1287 jours


def test_policy_on_new_data(Q, prices):
    day = 0
    shares = 0
    cash = INITIAL_CASH
    T = len(prices)

    while day < T - 1:
        s = get_Q_state(day, shares, cash)
        if s not in Q:
            a = 0
        else:
            a = np.argmax(Q[s])
        current_price = prices[day]
        if a == 1:  # Buy
            cost = SHARE_BATCH * current_price
            if cash >= cost and shares + SHARE_BATCH <= MAX_SHARES:
                shares += SHARE_BATCH
                cash -= cost
        elif a == 2:  # Sell
            if shares >= SHARE_BATCH:
                shares -= SHARE_BATCH
                cash += SHARE_BATCH * current_price
        day += 1

    final_value = get_portfolio_value(day - 1, shares, cash, prices)
    return final_value

final_value_ge = test_policy_on_new_data(Q, prices_ge)
print(f"[TEST SUR GE] Valeur finale du portefeuille (politique apprise sur BAC) : {final_value_ge:.2f}")


def simulate_policy_and_log(Q, prices):

    days_log = []
    actions_log = []
    portfolio_log = []
    
    day = 0
    shares = 0
    cash = INITIAL_CASH
    T = len(prices)
    
    while day < T - 1:
        s = get_Q_state(day, shares, cash)
        if s not in Q:
            a = 0  # par défaut, a = 0 (Hold)
        else:
            a = np.argmax(Q[s])
        
        days_log.append(day)
        actions_log.append(a)
        portfolio_log.append(get_portfolio_value(day, shares, cash, prices))
        
        current_price = prices[day]
        if a == 1:  # Buy
            cost = SHARE_BATCH * current_price
            if cash >= cost and shares + SHARE_BATCH <= MAX_SHARES:
                shares += SHARE_BATCH
                cash -= cost
        elif a == 2:  # Sell
            if shares >= SHARE_BATCH:
                shares -= SHARE_BATCH
                cash += SHARE_BATCH * current_price
        
        day += 1
    
    days_log.append(day)
    actions_log.append(np.nan)
    portfolio_log.append(get_portfolio_value(day - 1, shares, cash, prices))
    
    return days_log, actions_log, portfolio_log

days_ge, actions_ge, portfolio_ge = simulate_policy_and_log(Q, prices_ge)

# Marking :
days_buy = [d for d, a in zip(days_ge, actions_ge) if a == 1]
days_sell = [d for d, a in zip(days_ge, actions_ge) if a == 2]

# 3 subplots
plt.figure(figsize=(12, 10))

# Subplot 1 : Actions
plt.subplot(3, 1, 1)
days_hold = [d for d, a in zip(days_ge, actions_ge) if a == 0]
plt.scatter(days_hold, [0]*len(days_hold), color='blue', s=10, label='Hold')
plt.scatter(days_buy, [1]*len(days_buy), color='green', s=40, marker='^', label='Buy')
plt.scatter(days_sell, [2]*len(days_sell), color='red', s=40, marker='v', label='Sell')
plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
plt.xlabel("Jour")
plt.ylabel("Action")
plt.title("Actions au fil du temps")
plt.legend()
plt.grid(True)

# Subplot 2 : Portfolio evolution
plt.subplot(3, 1, 2)
plt.plot(days_ge, portfolio_ge, color='green', linewidth=1.5)
plt.xlabel("Jour")
plt.ylabel("Valeur du portefeuille (USD)")
plt.title("Évolution de la valeur du portefeuille")
plt.grid(True)

# Subplot 3 : GE Action
plt.subplot(3, 1, 3)
days_price = np.arange(len(prices_ge))
plt.plot(days_price, prices_ge, color='blue', linewidth=1.5, label='Prix GE')

plt.scatter(days_buy, prices_ge[days_buy], color='green', s=80, marker='^', label='Buy')
plt.scatter(days_sell, prices_ge[days_sell], color='red', s=80, marker='v', label='Sell')
plt.xlabel("Jour")
plt.ylabel("Prix de l'action (USD)")
plt.title("Évolution du prix de l'action GE (CSV)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
