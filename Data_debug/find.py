with open("data/brain.txt", "r", encoding="utf-8", errors="replace") as Him:
	for i, line in enumerate(Him):
		if '\x7f' in line:
			print(f"found at {i+1}: {line.strip()}")