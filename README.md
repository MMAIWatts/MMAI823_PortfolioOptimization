# MMAI823- assign1 -Stock Portfolio Optimization
  
### Introduction	
Finding a balance in anything allows for peak efficiency. In the context of stock and bond portfolios, finding the balance of the risk and the reward is critical to intelligent investing. This report outlines the process taken by Team Watts to assemble and optimize a portfolio to minimize its risk while maximizing returns.
Using historical stock price data from Yahoo Finance, we created an estimate of the variance and covariance between the 15 stocks in our portfolio. Beta values were scraped from the same webpage where possible. CAPM pricing model was used to fundamental methodology to estimate their expected returns. Using this information, we built a portfolio by optimizing return to risk trade-off by using Scipy’s Optimize solver. Furthermore, from that process, an optimal risk-reward point was picked from the efficiency frontier with an expected return of 10%.
 
### Stock Picks
Our investment horizon formed the basis for the rationale in constructing our portfolio. Team Watts consists of 5 young professionals with an expected investment horizon of 20-30 years. Our portfolio includes 15 stocks, primarily in the technology category.
Our team assumes that the overall economy will always grow eventually as productivity increases with technology advancement. With that assumption in mind, our team’s risk appetite is relatively high, resulting in a targeted beta in the range of 1.4 to 1.6 for our portfolio. Another assumption that we made as a team is that we believe in the next 20-30 years, the world will enter the next industrial revolution in automation, robotics, and AI. Hence our portfolio is constructed to gain broad exposure to this factor. 
Companies such as Lockheed Martin, Tesla, Amazon, and Rockwell Automation are all a part of our portfolio. Furthermore, our portfolio has a wide range of diversification within the technology category by investing in different technology industries. Our portfolio not only has exposure to the computing/software industry such as Alphabet, but it also has a considerable amount of exposure to biotech and industrial technology companies such as Thermo Fisher Scientifics, Caterpillar, and Bausch Health Companies.
From a risk management perspective, Team Watt’s portfolio focused primarily on large-cap stocks with (13 out of 15 stocks being large-cap) core technology and know-how. This diversification strategy ensures us (the investors) that in the event of financial mismanagement, there is a higher likelihood that any downfall of the companies in our portfolio would result in a buyout, merger, or acquisition.
Rebalancing this portfolio quarterly ensures the portfolio does not deviate from our target beta range and reduces exposures to high performing stocks. Additionally, it grants an opportunity for stocks that are currently under the EFT to be a part of the portfolio in the future.

Table 1: Selected Portfolio Makeup

Company Name | Symbol	Yahoo | Beta
------------ | ------------ | -------
Alphabet Inc.	|GOOG	|1.02
Nike Inc.|	NKE	|0.83
Lockheed Martin Corporation	|LMT	|0.95
Caterpillar	|CAT	|1.51
Tesla Inc.	|TSLA |	0.58
Amazon Inc.	|AMZN|	1.52
Thermo Fisher Scientific Inc. |	TMO |	1.12
Canadian Solar, Inc.	| CSIQ	| 2.04
Hewlett Packard Enterprise Company |	HPE	| 1.53
Micron Technology Inc. |	MU |	1.80 
Dell Technologies Inc.	| DELL	| 0.741
Advanced Micro Devices Inc.	| AMD |	3.08
Bausch Health Companies |	BHC	| 0.89
Rockwell Automation Inc.	| ROK |	1.44
Take-Two Interactive Software, Inc. |	TTWO	| 0.69

### Portfolio Optimization
Two frontier lines were plotted; one for the risk frontier and one that includes a risk-free option in the portfolio. The risk frontier is shown as a solid line in blue; the risk-free boundary is shown as a dashed line in green.
Additional simulated returns were used to illustrate how the frontier is genuinely the bounds of the random returns.


Figure 1: Efficiency Frontier Plot with simulated returns

![Efficiency Frontier](/graphs/EF_withRiskFree.png)

Our analysis found that some stocks had lower estimated returns with more risk (standard deviation), hence sub-optimal Sharpe Ratio. Expectedly, our optimization process attributed these following stocks with a weight of 0. These stocks are GOOG, NKE, TSLA, MU, DELL, BHC, TTWO.





