%matplotlib inline

import sys
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

"""
Environment definition:
- Iterative GSP auction environment, google adwords style
- One time iteration carries n_hour impressions, thus auctions with a fixed set of bids.
- Three ad slots, each slot has a constant probability of being clicked.
- Two other bidders in addition to the one we're acting as.
- Each bidder has a constant Qscore, the auction cpc is a bid of the next participant + reserved price.
- Each hour carries the same set of probabilities and traffic, no seasonality is present.
"""


n_hour = 100

cr = 0.1 # 2nd step: switch to Bernoulli process here
cv = 5 # 2nd step: switch to Gaussian here
reserved_price = 0.01
competitors_bids = np.array([0.1, 0.4])
ad_slot_ctrs = np.array([0.4, 0.2, 0.07])

# Just to start with, let's observe the process of flattenning of costs to the bid increase as well as inverse ROAS-to-bid curve
sim_metrics = {
    'bid': [],
    'cpc': [],
    'clicks': [],
    'total_cv': [],
    'position': []
}
sim_hours_per_bid=10
for our_bid in np.arange(0.01, 1, 0.02):
    for _ in range(sim_hours_per_bid):
        sim_metrics['bid'].append(our_bid)
        bids_below_us = competitors_bids[competitors_bids < our_bid]
        bids_above_us = competitors_bids[competitors_bids >= our_bid]
        our_cpc = (np.max(bids_below_us) if len(bids_below_us) > 0 else 0) + reserved_price
        our_position = len(bids_above_us)
        our_clicks = ad_slot_ctrs[our_position] * n_hour
        our_hour_cv = our_clicks * cr * cv
        sim_metrics['cpc'].append(our_cpc)
        sim_metrics['position'].append(our_position)
        sim_metrics['clicks'].append(our_clicks)
        sim_metrics['total_cv'].append(our_hour_cv)
        
sim_performance = pd.DataFrame(sim_metrics).assign(
    costs=lambda x: x.clicks*x.cpc,
    roas=lambda x: x.total_cv/x.costs,
    cpc_bid_diff=lambda x: x.bid - x.cpc,
    profit=lambda x: x.total_cv - x.costs
)
fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
sns.lineplot(data=sim_performance, x="bid", y="costs", ax=ax11)
sns.lineplot(data=sim_performance, x="bid", y="roas", ax=ax12)
sns.lineplot(data=sim_performance, x="bid", y="cpc_bid_diff", ax=ax21)
sns.lineplot(data=sim_performance, x="bid", y="profit", ax=ax22)
fig.set_size_inches(15, 15)