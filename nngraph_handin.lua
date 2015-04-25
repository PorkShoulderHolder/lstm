require "nngraph"


i1 = torch.Tensor{1,2,3,4}
i2 = torch.Tensor{4,3,5,6}
i3 = torch.Tensor{1,4,5,1,1,1,1,1}


x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Identity()()

lin = nn.Linear(8,4)(x3)
--linmodule = nn.gModule({x3},{lin})

--print(linmodule:forward(i3))

mult = nn.CMulTable()({x2, lin})
madd = nn.CAddTable()({x1, mult})
m = nn.gModule({x1,x2,x3}, {madd})
graph.dot(m.fg, 'MLP')
print(m:forward({i1,i2,i3}))

