import unittest
import decision_tree
import numpy as np

from decision_tree import Tree

x = np.load("object_result.dat")
y = np.load("types_result.dat")
trees = decision_tree.load_pickle('six_trees.pkl')
forest = decision_tree.load_pickle('random_forest.pkl')


class TestTree(unittest.TestCase):

    def test_remapping(self):
        binary_t = np.arange(1, 7)
        result = decision_tree.remapping(binary_t, 1)
        self.assertEqual(result, [1, 0, 0, 0, 0, 0])
        binary_t = [2, 2, 2, 2, 2]
        result = decision_tree.remapping(binary_t, 1)
        self.assertEqual(result, [0, 0, 0, 0, 0])

    def test_majority_value(self):
        targets = [1, 1, 1, 0, 0]
        value = decision_tree.majority_value(targets)
        self.assertEqual(value, 1)
        targets = [1, 1, 0, 0, 0]
        value = decision_tree.majority_value(targets)
        self.assertEqual(value, 0)

    def test_same_value(self):
        test = np.zeros(6)
        result = decision_tree.same_value(test)
        self.assertTrue(result)
        test = [1, 2, 3]
        result = decision_tree.same_value(test)
        self.assertFalse(result)

    def test_same_example(self):
        test = np.arange(1, 5)
        result = decision_tree.same_examples(test)
        self.assertFalse(result)
        test2 = np.matrix([test, test, test])
        result = decision_tree.same_examples(test2)
        self.assertTrue(result)
        test3 = np.array([1, 2, 3])
        result = decision_tree.same_examples(test3)
        self.assertFalse(result)

    def test_cal_entropy(self):
        result = decision_tree.calculate_entropy(2, 3)
        self.assertEqual(result, 0.9709505944546686)
        result = decision_tree.calculate_entropy(0, 5)
        self.assertEqual(result, 0)
        result = decision_tree.calculate_entropy(5, 0)
        self.assertEqual(result, 0)

    def test_cal_rmd(self):
        result = decision_tree.calculate_rmd(1, 2, 3, 4)
        self.assertEqual(result, 0.965148445440323)

    def test_room_classify(self):
        targets = [1, 0, 0, 0, 0]
        result = decision_tree.room_classify(targets)
        self.assertEqual(result, 1)
        '''decision_tree.room_classify([0, 0, 0, 0, 0])
        decision_tree.room_classify([0, 0, 0, 1, 1])'''

    def test_confusion_m(self):
        y = [1, 2, 2, 1, 1, 1]
        prediction = [1, 2, 1, 1, 2, 1]
        result = decision_tree.confusion_matrix(y, prediction)
        confusion_matrix = np.zeros(shape=(5, 5))
        confusion_matrix[0][0] = 3
        confusion_matrix[0][1] = 1
        confusion_matrix[1][0] = 1
        confusion_matrix[1][1] = 1
        np.testing.assert_array_equal(result, confusion_matrix)

    def test_recall_rate(self):
        y = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        prediction = [1, 2, 2, 3, 4, 5, 2, 3, 4, 5]
        con_m = decision_tree.confusion_matrix(y, prediction)
        result = decision_tree.recall_rate(con_m)
        true_value = [0.5, 1.0, 0.5, 0.5, 0.5]
        self.assertEqual(result, true_value)

    def test_precision_rate(self):
        y = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        prediction = [1, 2, 2, 3, 4, 5, 2, 3, 4, 5]
        con_m = decision_tree.confusion_matrix(y, prediction)
        result = decision_tree.precision_rate(con_m)
        true_value = [1.0, 0.6666666666666666, 0.5, 0.5, 0.5]
        self.assertEqual(result, true_value)

    def test_f_measure(self):
        result = decision_tree.f_measure(5, 9, 1)
        self.assertEqual(result, 6.428571428571429)

    def test_classification_rate(self):
        confusion_matrix = np.zeros(shape=(5, 5))
        confusion_matrix[0][0] = 3
        confusion_matrix[0][1] = 1
        confusion_matrix[1][0] = 1
        confusion_matrix[1][1] = 1
        result = decision_tree.classification_rate(confusion_matrix)
        self.assertEqual(result, 0.6666666666666666)

    def test_error_rate(self):
        y = [1, 2, 2, 1, 1, 1]
        prediction = [1, 2, 1, 1, 2, 1]
        result = decision_tree.error_rate(prediction, y)
        self.assertEqual(result, 0.3333333333333333)

    def test_plus_one(self):
        y = np.arange(1, 7)
        result = decision_tree.plus_one(y)
        np.testing.assert_array_equal(result, np.arange(2, 8))

    def test_find_leaf(self):
        leaf = decision_tree.find_leaf(trees[0], x[0])
        self.assertEqual(leaf, 1)

    def test_random_forest(self):
        prediction = decision_tree.test_forest(forest, x[0:3])
        y = np.ones(3)
        np.testing.assert_array_equal(y, prediction)

    def test_calculate_ig(self):
        ig = decision_tree.calculate_ig(x, 3, y)
        self.assertEqual(ig, 0.12419183456335703)

    def test_choose_best_attr(self):
        attri = np.arange(0, 57)
        best = decision_tree.choose_best_decision_attribute(x, attri, y)
        self.assertEqual(best, 14)

    def test_target_entropy(self):
        entropy = decision_tree.calculate_target_entropy(y)
        self.assertEqual(entropy, 0.9839393951635756)

    def test_init_tree(self):
        node = decision_tree.Tree()
        self.assertIsNone(node.op)
        self.assertIsNone(node.node_class)

    def test_find_new_example(self):
        example = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
        new_example = decision_tree.find_new_example(example, 0, 0)
        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
        np.testing.assert_array_equal(new_example, result)

    def test_find_new_target(self):
        example = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
        target = np.array([0, 1, 2, 3])
        new_example = decision_tree.find_new_target(example, target, 0, 0)
        result = np.array([0, 2, 3])
        np.testing.assert_array_equal(new_example, result)

    def test_delete_best_attr(self):
        attr = np.array([1, 2, 3, 4])
        new_attr = decision_tree.delete_best_attribute(attr, 3)
        result = np.array([1, 2, 4])
        np.testing.assert_array_equal(new_attr, result)

    def test_load_pickle(self):
        result = [1, 2, 3, 4]
        array = decision_tree.load_pickle('array.pkl')
        np.testing.assert_array_equal(array, result)

    def test_decision_tree_learning(self):
        attributes = np.arange(0, x[0].size)
        decision_tree.decision_tree_learning(x, attributes, y)

    def test_six_tree_creation(self):
        decision_tree.six_trees_creation(x, y)

    def test_random_tree(self):
        decision_tree.random_forest_creation(x, y, 2)

    def test_tree_evaluation(self):
        decision_tree.tree_evaluation(x, y, 2)


if __name__ == "__main__":
    unittest.main()
