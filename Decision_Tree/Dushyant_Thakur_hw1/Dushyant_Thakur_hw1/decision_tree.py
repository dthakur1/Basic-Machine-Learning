import math
import csv,collections
import sys

###########################################################################################################################################################################################    Node structure


class Node:
    def __init__(self):
        self.label = None
        self.name = None
        self.feature_number = None
        self.mode = None
        self.children = {}
        self.temp_children = None                                              ##To be used for making the node a leaf or node

    def make_leaf(self):
        if self.label is None:
            self.label = self.mode
            self.temp_children = self.children
            self.children = {}

    def make_node_again(self):                                                 ##To be used while pruning
        if self.label is not None:
            self.label = None
            self.children = self.temp_children
            self.temp_children = {}


    def num_nodes(self):
        acc = 1
        if self.children:
            children = self.children.values()
            for c in children:
                acc += c.num_nodes()

        return acc

    def classify(self, example):
        if self.label is not None:
            return self.label

        else:
            attr = self.feature_number
            instance_val = example[attr]

            try:
                next = self.children[instance_val]
                
                return next.classify(example)

            except (IndexError, KeyError):
                return self.mode

#####################################################################################################################################################################################    PRINT TREE
def print_tree(tree_root, tabs = 0):
    pre = "|    " * tabs
    output = ""

    if tree_root.label is not None:
        output += pre + "CLASS: " + str(tree_root.label) + "\n"


    else:
        for val, child in tree_root.children.items():
            output += "%s%s: %s" % (pre, tree_root.name, val) + "\n"
            output += print_tree(child, tabs + 1)
    return output

       


############################################################################################################################################################################################   ID3
def ID3(data_set, attribute_metadata):
   
    tree = Node()
    tree.mode = mode(data_set)
    
    homogenous = check_homogenous(data_set)
    attr = pick_best_attribute(data_set, attribute_metadata)
    if homogenous:
        tree.label = homogenous

    elif (not attr):
        tree.label = mode(data_set)
    
    else:
        tree.feature_number = attr 
        tree.name = attribute_metadata[attr]        
        temp_data = data_set

        children = {}

        partition = split_on_attribute(temp_data, attr)

        for value, examples in partition.items():
                children[value] = ID3(examples, attribute_metadata)

        
        tree.children = children

    return tree

def ID3_VI(data_set, attribute_metadata):
   
    tree = Node()
    tree.mode = mode(data_set)    
    homogenous = check_homogenous(data_set)

    attr = pick_best_attribute_VI(data_set, attribute_metadata)

    if homogenous:
        tree.label = homogenous

    elif (not attr):
        tree.label = mode(data_set)
    
    else:
        tree.feature_number = attr 
        tree.name = attribute_metadata[attr]        
        temp_data = data_set

        children = {}

        partition = split_on_attribute(temp_data, attr)

        for value, examples in partition.items():
                children[value] = ID3_VI(examples, attribute_metadata)        
        tree.children = children

    return tree


def check_homogenous(data_set):
    value = data_set[0][0]
    
    for x in data_set[1:]:
        if x[0] != value:
            return None

    return value


def mode(data_set):
    counts = [0, 0]

    for x in data_set:
        
        categorical = x[0]
        counts[categorical] += 1

    return int(not counts[0] > counts[1])


def entropy(data_set):
    total_examples = len(data_set)
    pos_examples = len([x for x in data_set if x[0] == 1])
    neg_examples = len([x for x in data_set if x[0] == 0])

    if total_examples == 0 or pos_examples == 0:
        pr_pos = 0
    else:
        pr_pos = float(pos_examples) / float(total_examples)

    if total_examples == 0 or neg_examples == 0:
        pr_neg = 0
    else:
        pr_neg = float(neg_examples) / float(total_examples)

    entropy_pos = -pr_pos * math.log(pr_pos, 2) if pr_pos > 0.0 else 0
    entropy_neg = -pr_neg * math.log(pr_neg, 2) if pr_neg > 0.0 else 0
#    print(entropy_pos + entropy_neg)
    return entropy_pos + entropy_neg

def variance_impurity(data_set):
    total_examples = len(data_set)
    pos_examples = len([x for x in data_set if x[0] == 1])
    neg_examples = len([x for x in data_set if x[0] == 0])

    if total_examples == 0 or pos_examples == 0:
        pr_pos = 0
    else:
        pr_pos = float(pos_examples) / float(total_examples)

    if total_examples == 0 or neg_examples == 0:
        pr_neg = 0
    else:
        pr_neg = float(neg_examples) / float(total_examples)
    return pr_pos*pr_neg
    
    
def split_on_attribute(data_set, attribute):
    partition = {}

    values = {x[attribute] for x in data_set}

    for value in values:
        partition[value] = [x for x in data_set if x[attribute] == value]

    return partition



def gain_entropy(data_set, attribute):
    
    current_entropy = entropy(data_set)
    total_examples = len(data_set)
    partition = split_on_attribute(data_set, attribute)

    entropy_after = 0
#    intrinsic_value = 0

    for subset in partition.values():
        if total_examples > 0:
            p = len(subset) / float(total_examples)  
            entropy_after += p * entropy(subset)

            # Compute partial intrinsic value
#            iv = -p * math.log(p, 2)
 #           intrinsic_value += iv

    info_gain = current_entropy - entropy_after
#    igr = info_gain / float(intrinsic_value) if intrinsic_value > 0 else 0

    return info_gain

def gain_VI(data_set, attribute):
    
    current_VI = variance_impurity(data_set)
    total_examples = len(data_set)
    partition = split_on_attribute(data_set, attribute)

    VI_after = 0
    
    for subset in partition.values():
        if total_examples > 0:
            p = len(subset) / float(total_examples)  
            VI_after += p * variance_impurity(subset)

    info_gain = current_VI - VI_after
    return info_gain


def pick_best_attribute(data_set, attribute_metadata):

    result = 0
    ig_max = 0

    for i, a in enumerate(attribute_metadata[1:]):
        i += 1 
        temp_data = data_set

        ig = gain_entropy(temp_data, i)
        
        if ig > ig_max:
            ig_max = ig
            result = i

    return result

def pick_best_attribute_VI(data_set, attribute_metadata):

    result = 0
    ig_max = 0

    for i, a in enumerate(attribute_metadata[1:]):
        i += 1 
        temp_data = data_set

        ig = gain_VI(temp_data, i)
        
        if ig > ig_max:
            ig_max = ig
            result = i

    return result


##########################################################################################################################################################################################    PARSE
def parse(filename):
    '''
    takes a filename and returns attribute information and all the data in array of arrays
    This function also rotates the data so that the 0 index is the winner attribute, and returns
    corresponding attribute metadata
    '''
    # initialize variables
    array = []
    attributes = ['CLASS','XB','XC','XD','XE','XF','XG','XH','XI','XJ','XK','XL','XM','XN','XO','XP','XQ','XR','XS','XT','XU']

    csvfile = open(filename,'r')
    fileToRead = csv.reader(csvfile, delimiter=' ',quotechar=',')
    # skip first line of data
    next(fileToRead)

    # set attributes
    
    # iterate through rows of actual data
    for row in fileToRead:
        # change each line of data into an array
        temp =row[0].split(',')
        # rotate data so that the target attribute is at index 0
        d = collections.deque(temp)
        d.rotate(1)
        array.append(list(d))
    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            temp = int(array[i][j])
            array[i][j] = temp
             
    return array, attributes
############################################################################################################################################################     ACCURACY
def validation_accuracy(tree,validation_set):

    total_instances = len(validation_set)
    accurate_instances = 0

    for x in validation_set:
        true_class = x[0]  
        computed_class = tree.classify(x)
        if computed_class == true_class:
            accurate_instances += 1

    accuracy = accurate_instances / float(total_instances)
    return accuracy 
###################################################################################################################################################################   PRUNE
def reduced_error_pruning(root, training_set, validation_set):
 
    accuracy = validation_accuracy(root, validation_set)
    nodes = [root]

    while len(nodes) > 0:
        n = nodes.pop()
        
        if n.label is None:
            # Temporarily make it a leaf to compute validation accuracy
            n.make_leaf()
            acc = validation_accuracy(root, validation_set)
            n.make_node_again()

            if acc >= accuracy:
                accuracy = acc
                n.make_leaf()
            else:
                try:
                    nodes.extend(x for x in n.children.values())
                
                except (KeyError, IndexError):
                    print("Error.")

    return root

###################################################################################################################################################################### DRIVER
def decision_tree_driver(train, validate, test, to_print, to_prune):
    
    train_set, attribute_metadata = parse(train)
    validate_set, _ = parse(validate)
    test_set, _ = parse(test)
    pruning_set, _ = parse(validate)
    
    
    print ("Training Tree for both heuristics")
    tree = ID3(train_set, attribute_metadata)
    tree2 = ID3_VI(train_set, attribute_metadata)
    print ('\n')
                                                                                    #############################################################################   ENTROPY HEURISTIC
    print('###\n#  Validating the entropy heuristic tree\n###')
    
    accuracy = validation_accuracy(tree, train_set)
    print("Accuracy on training set for entropy heuristic without pruning: " + str(accuracy))
    
    accuracy = validation_accuracy(tree,validate_set)
    print("Accuracy on validation set for entropy heuristic without pruning: " + str(accuracy))
    
    accuracy = validation_accuracy(tree,test_set)
    print("Accuracy on test set for entropy heuristic without pruning: " + str(accuracy))
    
    print('')
    if to_print and not to_prune:
        print("Tree without pruning the entropy heuritic tree")
        print(print_tree(tree))        

    if to_prune:
        print('###\n#  Pruning the entropy heuristic tree\n###')
        print("Nodes before pruning the entropy heuritic tree: ") 
        print(str(tree.num_nodes()))      
        tree = reduced_error_pruning(tree,train_set,pruning_set)
        print("Nodes after pruning the entropy heuritic tree: ") 
        print(str(tree.num_nodes()))      
        print('')

        accuracy = validation_accuracy(tree, train_set)
        print("Accuracy on training set for the entropy heuritic tree after pruning: " + str(accuracy))    
        accuracy = validation_accuracy(tree,validate_set)
        print("Accuracy on validation set for the entropy heuritic tree: " + str(accuracy))
        accuracy = validation_accuracy(tree,test_set)
        print("Accuracy on test set for the entropy heuritic tree: " + str(accuracy))


        print('')
        
        if to_print:
            print("Tree after pruning the entropy heuritic tree")
            print(print_tree(tree))
    
    
    
    #######################################################################################################################################################################
    print("########################################################################################################################################")
    print("########################################################################################################################################")

    print('###\n#  Validating the VI heuristic tree\n###')
    
    accuracy = validation_accuracy(tree2,train_set)
    print("Accuracy on training set for VI heuristic without pruning: " + str(accuracy))
    
    accuracy = validation_accuracy(tree2 ,validate_set)
    print("Accuracy on validation set for VI heuristic without pruning: " + str(accuracy))
    
    accuracy = validation_accuracy(tree2,test_set)
    print("Accuracy on test set for VI heuristic without pruning: " + str(accuracy))


    print('')
    if to_print and not to_prune:
        print("Tree without pruning the VI heuritic tree")
        print(print_tree(tree2))        

    if to_prune:
        print('###\n#  Pruning the VI heuristic tree\n###')
        print("Nodes before pruning the VI heuritic tree: ") 
        print(str(tree2.num_nodes()))      
        tree2 = reduced_error_pruning(tree2,train_set,pruning_set)
        print("Nodes after pruning the VI heuritic tree: ") 
        print(str(tree2.num_nodes()))      
        
        print('')
        
        accuracy = validation_accuracy(tree2, train_set)
        print("Accuracy on training set for the VI heuritic tree after pruning: " + str(accuracy))
        
        accuracy = validation_accuracy(tree2,validate_set)
        print("Accuracy on validation set for the VI heuritic tree after pruning: " + str(accuracy))
        
        accuracy = validation_accuracy(tree2,test_set)
        print("Accuracy on test set for the VI heuritic tree after pruning: " + str(accuracy))
        
        print('')
        if to_print:
            print("Tree after pruning the VI heuritic tree")
            print(print_tree(tree2))


############################################################################################    
    return tree,tree2
    
options = {
    'train' : sys.argv[1],
    'validate': sys.argv[2],
    'test': sys.argv[3],
    'to_print': sys.argv[4],
    'to_prune' : sys.argv[5],
}       
    
tree,tree2 = decision_tree_driver( **options )