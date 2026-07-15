it('should transform import.meta.webpackHot to false', () => {
	expect(import.meta.webpackHot).toBeUndefined();
	expect(import.meta["webpackHot"]).toBeUndefined();
	expect(typeof import.meta.webpackHot).toBe("undefined");

	let hot = false;
	if (import.meta.webpackHot) {
		hot = true;
    import.meta.webpackHot.accept();
  }

	expect(hot).toBe(false);
})
