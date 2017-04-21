# -*- coding: utf-8 -*-

import csv

from log_util import LogUtil
from singleton import Singleton


@Singleton
class LabelUtil:
    _log = None

    # dataPath
    def __init__(self):
        self._log = LogUtil().getlogger()
        self._log.debug("LabelUtil init")

    def __call__(self):
        print
        "called"

    def loadUnicodeSet(self, unicodeFilePath):
        self.byChar = {}
        self.byIndex = {}
        self.unicodeFilePath = unicodeFilePath

        with open(unicodeFilePath) as data_file:
            data_file = csv.reader(data_file, delimiter=',')

            self.count = 0
            for r in data_file:
                self.byChar[r[0]] = int(r[1])
                self.byIndex[int(r[1])] = r[0]
                self.count += 1

    def toUnicode(self, src, index):
        # 1 byte
        code1 = int(ord(src[index + 0]))

        index += 1

        result = code1

        return result, index

    def convertWordToGrapheme(self, label):

        result = []

        index = 0
        while index < len(label):
            (code, nextIndex) = self.toUnicode(label, index)

            result.append(label[index])

            index = nextIndex

        return result, "".join(result)

    def convertWordToNum(self, word):
        try:
            labelList, _ = self.convertWordToGrapheme(word)

            labelNum = []

            for char in labelList:
                # skip word
                if char == "":
                    pass
                else:
                    labelNum.append(int(self.byChar[char]))

            # tuple typecast: read only, faster
            return tuple(labelNum)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def convertNumToWord(self, numList):
        try:
            labelList = []
            for num in numList:
                labelList.append(self.byIndex[num])

            return ''.join(labelList)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def getCount(self):
        try:
            return self.count

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

    def getUnicodeFilePath(self):
        try:
            return self.unicodeFilePath

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

    def getBlankIndex(self):
        return self.byChar["-"]

    def getSpaceIndex(self):
        return self.byChar["$"]
