import numpy as np

from collections import Counter

class Variable:
    def __init__(self, name, domain, size):
        '''Create a variable object, specifying its name (a
        string). Optionally specify the initial domain.
        '''
        self.name = name                #text name for variable
        self.dom = list(domain)         #Make a copy of passed domain
        self.size = size
        self.evidence_index = 0         #evidence value (stored as index into self.dom)
        self.assignment_index = 0       #For use by factors. We can assign variables values
                                        #and these assigned values can be used by factors
                                        #to index into their tables.

class Factor:
    def __init__(self, name, scope, domain, values):
        '''create a Factor object, specify the Factor name (a string)
        and its scope (an ORDERED list of variable objects).'''
        self.scope = list(scope)
        self.name = name
        self.domain = domain
        self.values = values

class BayesianNetwork:
    def __init__(self, filename):
        f = open(filename, 'r') 
        N = int(f.readline())
        lines = f.readlines()

        variables = []
        factors = []
        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(line)
            # YOUR CODE HERE
            var = Variable(node, domain, shape[-1] if type(shape) == tuple else shape)
            variables.append(var)
            
            name_fact = ''
            scope_fact = []
            domain_fact = []
            values_fact = []
            for p in parents :
                name_fact += p + '_' 
                domain_fact.append(p)
                for v in variables:
                    if(v.name == p):
                        scope_fact.append(v)
            name_fact += node
            domain_fact.append(node)
            scope_fact.append(var)
            n = len(probabilities)
            n1 = len(scope_fact)
            tmp = [0]*n1
            for i in range(n): 
                size = n
                value = []
                for j in range(n1):
                    size = size / scope_fact[j].size
                    k = int(tmp[j]/size) % scope_fact[j].size
                    value.append(scope_fact[j].dom[k])
                    tmp[j] += 1
                value.append(probabilities[i])
                values_fact.append(value)

            factor = Factor(name_fact, scope_fact, domain_fact, values_fact)
            factors.append(factor)

        self.variables = variables
        self.factors = factors
        f.close()

    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        self.deductFactorByEvidence(evidence_variables)
        self.removeVariables(query_variables, evidence_variables)
        self.productAllFactors()
        alpha = self.calculate_alpha()
        result = self.final_result(query_variables, alpha)

        f.close()
        return result

    def deductFactorByEvidence(self, evidence):
        keys = list(evidence.keys())
        for key in keys:
            for factor in self.factors:
                if key in factor.domain:
                    i = factor.domain.index(key)
                    factor.values = list(filter(lambda x : x[i] == evidence[key] , factor.values))

    def removeVariables(self, query, evidence):
        X = set()
        for var in self.variables:
            X.add(var.name)
        Z = X - set(query.keys()) - set(evidence.keys())
        for z in Z:
            index = []
            for i,factor in enumerate(self.factors):
                if z in factor.domain:
                    index.append(i)
            if(len(index) == 1):
                # remove domain
                self.remove_domain(index[0], z)
                # sum_out var
                self.sum_out(self.factors[index[0]])
                # remove old factor
                newFactor = self.factors[index[0]]
                self.factors.remove(newFactor)
                # add new factor into head 
                self.factors.insert(0, newFactor)
            else:
                newFactor = self.multiply_factor(self.factors[index[0]], self.factors[index[1]], z)
                
                self.factors.pop(index[0])
                self.factors.pop(index[1] - 1)
                count = 2
                if (len(index) > 2):
                    for i in range(2, len(index)):
                        newFactor = self.multiply_factor(newFactor, self.factors[index[i]-count], z)
                        self.factors.pop(index[i] - count)
                        count += 1
                
                self.factors.insert(0, newFactor)
                # remove domain
                self.remove_domain(0, z)
                # sum_out var
                self.sum_out(self.factors[0])
            
                
    def remove_domain(self, index, z):
        i = self.factors[index].domain.index(z)
        self.factors[index].domain.remove(z)
        for value in self.factors[index].values:
            value.pop(i)
    
    def sum_out(self, factor):
        for i, value in enumerate(factor.values):
            for j, value1 in enumerate(factor.values[i+1:]):
                if(value[:-1] == value1[:-1]):
                    value[-1] += value1[-1]
                    i = len(factor.values) - factor.values[::-1].index(value1) - 1
                    factor.values.pop(i)
        

    def multiply_factor(self, factor1, factor2, z):
        index1 = factor1.domain.index(z)
        index2 = factor2.domain.index(z)
        name = factor1.name + '_' + factor2.name
        domain = factor1.domain + factor2.domain
        i = len(domain) - domain[::-1].index(z) - 1
        domain.pop(i)
        values = []
        for value1 in factor1.values:
            for value2 in factor2.values:
                if value1[index1] == value2[index2]:
                    num = value1[-1] * value2[-1]
                    temp_j = value2.copy()
                    temp_j.pop(index2)
                    temp = value1[:-1] + temp_j[:-1] + [num]
                    values.append(temp)
        newFactor = Factor(name, [], domain, values)
        return newFactor

    def productAllFactors(self):
        if len(self.factors) == 1:
            self.factors = self.factors[0]
            return
        if len(self.factors) == 2:
            common = list(set(self.factors[0].domain).intersection(self.factors[1].domain))
            new_factor = self.multiply_factor(self.factors[0], self.factors[1], common[0])
            self.factors = new_factor
            return
        for i in range(len(self.factors)-1):
            for j in range(i+1,len(self.factors)):
                common = list(set(self.factors[i].domain).intersection(self.factors[j].domain))
                if len(common) != 0:
                    new_factor = self.multiply_factor(self.factors[i], self.factors[j], common[0])
                    self.factors.remove(self.factors[i])
                    j -= 1
                    self.factors.remove(self.factors[j])
                    self.factors = [new_factor] + self.factors
                    return self.productAllFactors()
    
    def calculate_alpha(self):
        alpha = 0.0
        for value in self.factors.values:
            alpha += value[-1]
        return alpha

    def final_result(self, query, alpha):
        keys = list(query.keys())
        values = list(query.values())
        index = []
        for key in keys:
            index.append(self.factors.domain.index(key))
        result = 0.0
        for i, value in enumerate(self.factors.values):
            check = True
            for j, val_index in enumerate(index):
                if(value[val_index] != values[j]):
                    check = False
            if(check):
                result += value[-1]
        result = result/alpha
        return result

    def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        # YOUR CODE HERE


        f.close()
        return result

    def __extract_model(self, line):
        parts = line.split(';')
        node = parts[0]
        if parts[1] == '':
            parents = []
        else:
            parents = parts[1].split(',')
        domain = parts[2].split(',')
        shape = eval(parts[3])
        # probabilities = np.array(eval(parts[4])).reshape(shape)
        probabilities = np.array(eval(parts[4]))
        return node, parents, domain, shape, probabilities

    def __extract_query(self, line):
        parts = line.split(';')

        # extract query variables
        query_variables = {}
        for item in parts[0].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            query_variables[lst[0]] = lst[1]

        # extract evidence variables
        evidence_variables = {}
        for item in parts[1].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            evidence_variables[lst[0]] = lst[1]
        return query_variables, evidence_variables
