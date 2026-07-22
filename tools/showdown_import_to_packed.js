#!/usr/bin/env node
/** Convert normal Pokemon Showdown Import/Export files to packed team lines.
 *
 * Each input file can be one team or a Showdown backup-style collection with
 * `=== team ===` headings. The output is directly consumable by
 * convert_battle_factory_teams.py, which pairs adjacent lines into states.
 * Parsing and packing are delegated to a local Pokemon Showdown checkout so
 * this bridge follows Showdown's canonical format rather than reimplementing
 * it.
 *
 * Usage:
 *   node tools/showdown_import_to_packed.js --output packed.txt team-a.txt team-b.txt
 *   node tools/showdown_import_to_packed.js --showdown ../pokemon-showdown \
 *     --output packed.txt team-a.txt team-b.txt
 */

'use strict';

const fs = require('fs');
const path = require('path');

function usage(exitCode = 0) {
	const stream = exitCode ? process.stderr : process.stdout;
	stream.write(
		'Usage: node tools/showdown_import_to_packed.js ' +
			'[--showdown PATH] --output FILE TEAM.txt [TEAM.txt ...]\n' +
		'\nInputs may be one team each or collections separated by === headings.\n'
	);
	process.exit(exitCode);
}

function fail(message) {
	process.stderr.write(`error: ${message}\n`);
	process.exit(1);
}

function parseArgs(argv) {
	const options = {
		showdown: path.resolve(__dirname, '..', '..', 'pokemon-showdown'),
		output: '',
		inputs: [],
	};
	for (let index = 0; index < argv.length; index++) {
		const arg = argv[index];
		if (arg === '--help' || arg === '-h') usage();
		if (arg === '--showdown') {
			if (++index >= argv.length) fail('--showdown requires a path');
			options.showdown = path.resolve(argv[index]);
		} else if (arg === '--output' || arg === '-o') {
			if (++index >= argv.length) fail(`${arg} requires a path`);
			options.output = path.resolve(argv[index]);
		} else if (arg.startsWith('-')) {
			fail(`unknown option ${arg}`);
		} else {
			options.inputs.push(path.resolve(arg));
		}
	}
	if (!options.output) fail('--output is required');
	if (!options.inputs.length) fail('at least one team file is required');
	if (options.inputs.includes(options.output)) fail('output must not overwrite an input file');
	return options;
}

function loadTeams(showdownPath) {
	const modulePath = path.join(showdownPath, 'dist', 'sim');
	if (!fs.existsSync(path.join(showdownPath, 'package.json'))) {
		fail(`Pokemon Showdown checkout not found at ${showdownPath}`);
	}
	try {
		return require(modulePath).Teams;
	} catch (error) {
		fail(
			`could not load ${modulePath}; build the Showdown checkout with ` +
			`"npm run build" (${error.message})`
		);
	}
}

function validateTeam(team, inputPath) {
	if (!Array.isArray(team) || team.length !== 6) {
		fail(`${inputPath}: expected exactly six Pokemon, got ${team?.length ?? 0}`);
	}
	for (const [index, set] of team.entries()) {
		const label = set.name || set.species || `slot ${index + 1}`;
		if (!set.species) fail(`${inputPath}: ${label} has no species`);
		if (!Array.isArray(set.moves) || set.moves.length !== 4) {
			fail(`${inputPath}: ${label} must have exactly four moves`);
		}
	}
}

function splitExports(source, inputPath) {
	const lines = source.replace(/\r\n?/g, '\n').split('\n');
	const headings = [];
	for (const [index, line] of lines.entries()) {
		if (/^===.*===$/.test(line.trim())) headings.push(index);
	}
	if (!headings.length) return [{label: inputPath, source}];

	const sections = [];
	for (const [sectionIndex, start] of headings.entries()) {
		const end = headings[sectionIndex + 1] ?? lines.length;
		const label = `${inputPath} (${lines[start].trim()})`;
		const section = lines.slice(start + 1, end).join('\n').trim();
		if (!section) fail(`${label}: empty team section`);
		sections.push({label, source: section});
	}
	return sections;
}

function main() {
	const options = parseArgs(process.argv.slice(2));
	const Teams = loadTeams(options.showdown);
	const packed = options.inputs.flatMap(inputPath => {
		let source;
		try {
			source = fs.readFileSync(inputPath, 'utf8');
		} catch (error) {
			fail(`${inputPath}: ${error.message}`);
		}
		return splitExports(source, inputPath).map(section => {
			const team = Teams.import(section.source);
			validateTeam(team, section.label);
			return Teams.pack(team);
		});
	});

	fs.mkdirSync(path.dirname(options.output), {recursive: true});
	fs.writeFileSync(options.output, `${packed.join('\n')}\n`, 'utf8');
	process.stdout.write(`Converted ${packed.length} Showdown exports to ${options.output}\n`);
}

main();
