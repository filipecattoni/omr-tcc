def segment_staves(img, grouped_staff_rows):

	# deve retornar uma lista de pares com os valores de inicio e fim de cada pentagrama

	staves_n = int(len(grouped_staff_rows)/5)

	bounds = []

	upper = 0
	lower = int((grouped_staff_rows[5][0] + grouped_staff_rows[4][-1])/2)
	bounds.append([upper, lower])

	for i in range(1, staves_n-1):
		upper = lower
		lower = int((grouped_staff_rows[i*5+5][0] + grouped_staff_rows[i*5+4][-1])/2)
		bounds.append([upper, lower])

	upper = lower
	lower = len(img)
	bounds.append([upper, lower])

	return bounds