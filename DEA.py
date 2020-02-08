### Data envelopment analysis

import pulp as pp
import numpy as np


### Optimization problem
def LOP(input, output, i, RTS):
    prob = pp.LpProblem("DEA", pp.LpMaximize)
    u = pp.LpVariable('u', 0, None)
    v = pp.LpVariable('v', 0, None)
    if RTS == 'crs':
        prob += u * output[:, i]
        prob += (v * input[:, i]) == 1
        for j in range(0, len(output[0])):
            prob += (u * output[:, j]) <= (v * input[:, j])
        # Return efficiency
        prob.solve()
        return output[:, i] * u.varValue
    elif RTS == 'vrs':
        us = pp.LpVariable('us', None, None)
        prob += (u * output[:, i]) - us
        prob += v * input[:, i] == 1
        for j in range(0, len(output[0])):
            prob += (u * output[:, j]) <= (v * input[:, j]) + us
        # Return efficiency
        prob.solve()
        return (output[:, i] * u.varValue) - us.varValue
    else:
        raise Exception(ValueError, "Unknown return to scale model")


### DEA function
def DEA(input, output, RTS):
    if len(input[0]) != len(output[0]):
        raise Exception("Some DMU's output or input is missing! Check you data.")

    result = []
    # get all efficiencies
    for k in range(0, len(input[0])):
        result.append(round(LOP(input=input, output=output, i=k, RTS=RTS)[0], 4))
    return result


# Test - Compare with R
x = np.array([(1, 2, 6, 4, 5)])
y = np.array([(10, 100, 75, 100, 100)])
result_crs = []
result_crs = DEA(input=x, output=y, RTS='crs')
print(result_crs)
result_vrs = []
result_vrs = DEA(input=x, output=y, RTS='vrs')
print(result_vrs)
