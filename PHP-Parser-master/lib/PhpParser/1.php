<?php
use PhpParser\Error;
use PhpParser\NodeDumper;
use PhpParser\ParserFactory;

$code = <<<'CODE'
<?php

function test($foo)
{
    var_dump($foo);
}
CODE;

$parser = (new ParserFactory())->createForNewestSupportedVersion();
try {
    $ast = $parser->parse($code);
} catch (Error $error) {
    echo "Parse error: {$error->getMessage()}\n";
    return;
}

$dumper = new NodeDumper;
echo $dumper->dump($ast) . "\n";
