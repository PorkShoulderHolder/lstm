require "nngraph"

torch.manualSeed(1)

function create_module()
	local x1 = nn.Identity()()
	local x2 = nn.Identity()()
	local x3 = nn.Identity()()

	local lin = nn.Linear(8,4)(x3)
	local mult = nn.CMulTable()({x2, lin})
	local madd = nn.CAddTable()({x1, mult})
	local m = nn.gModule({x1, x2, x3}, {madd})

	return m
end

function test_module(m)

	i1 = torch.Tensor{4,3,5,6}
	i2 = torch.Tensor{1,2,3,4}
	i3 = torch.Tensor{1,4,5,1,1,1,1,1}
	print(m:forward({i2, i1,i3}))

m = create_module()
test_module(m)