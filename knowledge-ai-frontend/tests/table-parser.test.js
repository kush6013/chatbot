'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

global.window = {};
global.document = {
  addEventListener() {},
  createElement() {
    throw new Error('DOM element created at module load');
  },
  createTextNode() {
    throw new Error('DOM text node created at module load');
  }
};

const code = fs.readFileSync(
  path.join(__dirname, '..', 'app.js'),
  'utf8'
);
vm.runInThisContext(code);

const {
  splitMarkdownRow,
  isTableRow,
  isTableSeparator
} = window.AppMarkdown;

let passed = 0;
function test(name, fn) {
  fn();
  passed += 1;
  console.log(`ok - ${name}`);
}

test('legacy mangled header splits without merging columns', () => {
  assert.deepStrictEqual(
    splitMarkdownRow('| **DateFestival/HolidayDay** | | |'),
    ['**DateFestival/HolidayDay**']
  );
});

test('legacy mangled separator is recognised as a separator', () => {
  assert.strictEqual(
    isTableSeparator('| --------------------------- | -- | -- |'),
    true
  );
});

test('clean pipe header splits into its columns', () => {
  assert.deepStrictEqual(
    splitMarkdownRow('| Date | Festival/Holiday | Day |'),
    ['Date', 'Festival/Holiday', 'Day']
  );
});

test('clean separator row detected', () => {
  assert.strictEqual(isTableSeparator('| --- | --- | --- |'), true);
});

test('centered separator variant detected', () => {
  assert.strictEqual(isTableSeparator('|:---:|:---:|:---:|'), true);
});

test('data row starting with pipe is a table row', () => {
  assert.strictEqual(
    isTableRow('| 15 Aug 2026 | Independence Day | Saturday |'),
    true
  );
});

test('plain paragraph is not a table row', () => {
  assert.strictEqual(
    isTableRow('Work from home policy is only offered case by case.'),
    false
  );
});

test('trailing-pipe short row keeps its cells', () => {
  assert.deepStrictEqual(
    splitMarkdownRow('| 15 Aug 2026 | Independence Day |'),
    ['15 Aug 2026', 'Independence Day']
  );
});

test('no-prefix pipe row with enough pipes is a table row', () => {
  assert.strictEqual(
    isTableRow('15 Aug 2026 | Independence Day | Saturday |'),
    true
  );
});

console.log(`\n${passed} tests passed`);