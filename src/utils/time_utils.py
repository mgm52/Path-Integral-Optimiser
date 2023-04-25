import time

class TimeTester():
    def __init__(self,name,disabled=False):
        self.disabled=disabled
        if self.disabled: return
        self._start_time=time.time()
        self._start_dict={}
        self._stop_dict={}
        self.name=name
    def start(self,name):
        if self.disabled: return
        self._start_dict[name]=time.time()
        self.prev = name
    def end(self,name):
        if self.disabled: return
        self._stop_dict[name]=time.time()-self._start_dict[name]
    def end_prev(self):
        if self.disabled: return
        self._stop_dict[self.prev]=time.time()-self._start_dict[self.prev]
    def end_all(self):
        if self.disabled: return
        for key in self._start_dict.keys():
            self._stop_dict[key]=time.time()-self._start_dict[key]

        total = time.time()-self._start_time
        print(f"Time summary for {self.name}:")
        print(f"    Total time: {total}")
        for key in self._stop_dict.keys():
            print(f"    {key}: {self._stop_dict[key]/total*100:.2f}%")
        sub_total = sum(self._stop_dict.values())
        print(f"    Other: {(total-sub_total)/total*100:.2f}%")