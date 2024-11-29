class CyclicIterator:
    def __init__(self, obj, max_repeat=5):
        self.__obj = obj
        self.__max_repeat = max_repeat
        self.__current_repeat = 1
        self.__index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.__index < len(self.__obj):
            item = self.__obj[self.__index]
        else:
            if self.__current_repeat < self.__max_repeat:
                self.__current_repeat = self.__current_repeat + 1
                self.__index = 0
                item = self.__obj[self.__index]
            else:
                raise StopIteration()
        self.__index = self.__index + 1
        return item
