#!/usr/bin/env python3
"""
Database Dependency Analysis Tool - Main CLI Interface

This is the main entry point for the database dependency analysis tool,
providing command-line interface for all analysis operations.
"""

import argparse
import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import signal
from tqdm import tqdm
import pandas as pd
import yaml
from enum import Enum  # <-- Add this import

# Import all our modules
from config.config_loader import ConfigLoader, ConfigError
from parsers.sql_parser_ob import SQLParser




class DependencyAnalysisCLI:
    """Main command-line interface for the dependency analysis tool."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.config = None
        self.storage = None
        self.start_time = None
        self.interrupted = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self, level: str = 'INFO') -> logging.Logger:
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'sql_dependency_parser\\logs\\dependency_analysis_{datetime.now():%Y%m%d_%H%M%S}.log')
            ]
        )
        return logging.getLogger('DependencyAnalysisCLI')
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.interrupted = True
        # Save progress if analyzing
        if hasattr(self, '_save_progress'):
            self._save_progress()
        sys.exit(0)
    
    def main(self):
        """Main entry point."""
        print("Script started!")  # Add this line for debugging
        parser = self._create_parser()
        args = parser.parse_args()
        print(f"Arguments parsed: {args}")  # Add this line for debugging
        # ... rest of the method
        parser = self._create_parser()
        args = parser.parse_args()
        
        # Update logging level if specified
        if hasattr(args, 'log_level') and args.log_level:
            self.logger.setLevel(getattr(logging, args.log_level.upper()))
        
        # Execute command
        if hasattr(args, 'func'):
            try:
                args.func(args)
            except KeyboardInterrupt:
                self.logger.error("Operation interrupted by user")
                sys.exit(1)
            except Exception as e:
                self.logger.error(f"Fatal error: {str(e)}", exc_info=args.debug if hasattr(args, 'debug') else False)
                sys.exit(1)
        else:
            parser.print_help()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all commands."""
        parser = argparse.ArgumentParser(
            prog='db-dependency-analyzer',
            description='Analyze database dependencies in large codebases',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Analyze a repository
  %(prog)s analyze --repo-path /path/to/repo --output results/

  # Launch interactive dashboard
  %(prog)s visualize --data results/

  # Validate configuration
  %(prog)s validate --config config/
  
            """
        )
        
        # Global options
        parser.add_argument('-v', '--verbose', action='store_true',
                          help='Enable verbose output')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug mode with full stack traces')
        parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          default='INFO', help='Set logging level')
        
        # Create subparsers
        subparsers = parser.add_subparsers(title='commands', dest='command',
                                         help='Available commands')
        
        # Analyze command
        self._add_analyze_parser(subparsers)
        
        # Validate command
        self._add_validate_parser(subparsers)
        
        return parser
    
    def _add_analyze_parser(self, subparsers):
        """Add analyze command parser."""
        parser = subparsers.add_parser(
            'analyze',
            help='Analyze repository for database dependencies',
            description='Perform full analysis of database dependencies in source code'
        )
        
        # Required arguments
        parser.add_argument('--repo-path', required=True, type=Path,
                          help='Path to source repository')
        parser.add_argument('--output', required=True, type=Path,
                          help='Output directory for results')
        parser.add_argument('--batch-size', type=int, default=100,
                  help='Files per processing batch (default: 100)')
        parser.add_argument('--parallel', type=int, default=mp.cpu_count()-1,
                  help='Number of parallel processes')
        parser.add_argument('--memory-limit', type=int, default=4096,
                  help='Memory limit in MB (default: 4096)')
        
        # Optional arguments
        parser.add_argument('--config', type=Path, default=Path('config'),
                          help='Configuration directory (default: config)')
        parser.add_argument('--languages', nargs='+', 
                          choices=['SQL', 'PL/SQL', 'XML'],
                          help='Specific languages to process')
        
        parser.set_defaults(func=self.analyze_command)
    
    def _add_validate_parser(self, subparsers):
        """Add validate command parser."""
        parser = subparsers.add_parser(
            'validate',
            help='Validate configuration and test parsing',
            description='Verify configuration files and test parsers'
        )
        
        parser.add_argument('--config', type=Path, default=Path('config'),
                          help='Configuration directory')
        parser.add_argument('--test-file', type=Path,
                          help='Test parsing on specific file')
        parser.add_argument('--test-language', choices=['SQL', 'PL/SQL', 'XML'],
                          help='Language for test file')
        
        parser.set_defaults(func=self.validate_command)
    
    
    def analyze_command(self, args):
        """Execute analysis command."""
        self.logger.info("="*60)
        self.logger.info("Starting Database Dependency Analysis")
        self.logger.info("="*60)
        self.start_time = time.time()
        
        # Load configuration
        self.logger.info("Loading configuration...")
        try:
            self.config = ConfigLoader(args.config)
        except ConfigError as e:
            self.logger.error(f"Configuration error: {e}")
            sys.exit(1)
        
        # Validate configuration
        validation_result = self.config.validate_all()
        if not validation_result.is_valid:
            self.logger.error("Configuration validation failed:")
            for error in validation_result.errors:
                self.logger.error(f"  - {error}")
            sys.exit(1)

        # Scan repository
        self.logger.info(f"Scanning repository: {args.repo_path}")
        files_to_process = self._scan_repository(args.repo_path, args.languages)
        
        if not files_to_process:
            self.logger.warning("No files found to process")
            return
        
        self.logger.info(f"Found {len(files_to_process)} files to process")
        
        # Check memory
        self._check_memory(args.memory_limit)
        
        # Process files
        self.logger.info("Processing files...")
        parse_results = self._process_files(
            files_to_process,
            args.batch_size,
            args.parallel
        )
        
        # Save parsed data
        self.logger.info("Saving parsed data...")
        self._save_parsed_data(parse_results, args.output)
        
        """
        # Dependency analysis
        if not args.skip_analysis:
            self.logger.info("Analyzing dependencies...")
            self._analyze_dependencies()
        """
      
       
        elapsed = time.time() - self.start_time
        self.logger.info("="*60)
        self.logger.info(f"Analysis completed in {timedelta(seconds=int(elapsed))}")
        self.logger.info("="*60)
    
    def validate_command(self, args):
        """Execute validate command."""
        self.logger.info("Validating configuration...")
        
        # Load and validate configuration
        try:
            config = ConfigLoader(args.config)
            validation_result = config.validate_all()
            
            if validation_result.is_valid:
                self.logger.info("✓ Configuration is valid")
            else:
                self.logger.error("✗ Configuration validation failed:")
                for error in validation_result.errors:
                    self.logger.error(f"  - {error}")
            
            if validation_result.warnings:
                self.logger.warning("Warnings:")
                for warning in validation_result.warnings:
                    self.logger.warning(f"  - {warning}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
        

    
    def _scan_repository(self, repo_path: Path, languages: Optional[List[str]] = None) -> List[Path]:
        """Scan repository for files to process."""
        files = []
        exclusions = self.config.get_exclusion_patterns()
        
        # Get file patterns for specified languages
        if languages:
            extensions = []
            for lang in languages:
                extensions.extend(self.config.get_file_extensions(lang))
        else:
            # Get all configured extensions
            extensions = []
            for lang in self.config.get('parsing_rules.languages', {}).keys():
                extensions.extend(self.config.get_file_extensions(lang))
        
        # Scan files
        with tqdm(desc="Scanning files") as pbar:
            for ext in extensions:
                for file_path in repo_path.rglob(f"*{ext}"):
                    pbar.update(1)
                    
                    # Check exclusions
                    if self._is_excluded(file_path, exclusions):
                        continue
                    
                    # Check file size
                    if file_path.stat().st_size > self.config.get('global.max_file_size_mb', 100) * 1024 * 1024:
                        self.logger.warning(f"Skipping large file: {file_path}")
                        continue
                    
                    files.append(file_path)
                    

        
        return files
    
    def _is_excluded(self, file_path: Path, exclusions: Dict) -> bool:
        """Check if file should be excluded."""
        # Check directory exclusions
        for pattern in exclusions.get('directories', {}).get('patterns', []):
            if pattern.replace('**/', '') in str(file_path):
                return True
        
        # Check file exclusions
        for pattern in exclusions.get('files', {}).get('patterns', []):
            if file_path.match(pattern):
                return True
        
        return False
    
    def _check_memory(self, limit_mb: int):
        """Check available memory."""
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        
        if available_mb < limit_mb:
            self.logger.warning(
                f"Low memory warning: {available_mb:.0f}MB available, "
                f"{limit_mb}MB recommended"
            )
    
    def _process_files(self, files: List[Path], batch_size: int, 
                      parallel: int) -> List[Dict]:
        """Process files in parallel batches."""
        results = []
        processed = 0
        errors = 0
        
        # Create parser instances
        parsers = {
            'SQL': SQLParser(self.config.get_language_config('SQL')),
        }
        
        # Process in batches
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            # Submit batches
            futures = []
            for i in range(0, len(files), batch_size):
                batch = files[i:i+batch_size]
                self.logger.info(f"Submitting batch {i//batch_size+1} with {len(batch)} files")
                future = executor.submit(self._process_batch, batch, parsers)
                futures.append(future)
            
            # Process results
            with tqdm(total=len(files), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        processed += len(batch_results)
                        pbar.update(len(batch_results))
                        
                        # Check for interruption
                        if self.interrupted:
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Batch processing error: {e}")
                        errors += 1
        
        self.logger.info(f"Processed {processed} files with {errors} errors")
        return results
    
    def _process_batch(self, files: List[Path], parsers: Dict) -> List[Dict]:
        """Process a batch of files."""
        results = []
        logger = logging.getLogger('DependencyAnalysisCLI.Batch')
        logger.info(f"Starting batch with {len(files)} files")
        for file_path in files:
            logger.info(f"Processing file: {file_path}")
            try:
                # Determine language
                ext = file_path.suffix
                language = self._get_language_for_extension(ext)

                if language and language in parsers:
                    parser = parsers[language]
                    result = parser.parse_file(file_path)
                    logger.info(f"Finished file: {file_path}")

                    # Unpack result
                    objects_defined_dicts = []
                    for db_obj in result.objects_defined:
                        obj_dict = {
                            'name': db_obj.name,
                            'schema': db_obj.schema,
                            'object_type': db_obj.object_type.value,  # Convert Enum to string
                            'file_path': db_obj.file_path,
                            'start_position': db_obj.start_position,
                            'end_position': db_obj.end_position,
                            'start_line': db_obj.start_line,
                            'end_line': db_obj.end_line,
                            'full_content': db_obj.full_content,
                            'body_content': db_obj.body_content,
                            # Convert dependencies to dictionaries
                            'dependencies': [
                                {
                                    'target_name': dep.target_name,
                                    'target_schema': dep.target_schema,
                                    'target_object_type': dep.target_object_type.value,  # Convert Enum
                                    'dependency_type': dep.dependency_type.value,        # Convert Enum
                                    'line_number': dep.line_number,
                                    'context': dep.context
                                }
                                for dep in db_obj.dependencies
                            ]
                        }
                        objects_defined_dicts.append(obj_dict)
                    
                    # Add result to batch
                    results.append({
                        'file_path': str(file_path),
                        'language': language,
                        'objects_defined': objects_defined_dicts,
                        'errors': result.errors,
                        'warnings': result.warnings,
                        'parse_time': result.parse_time
                    })
                else:
                    logger.warning(f"Unknown language for file: {file_path}")
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}", exc_info=True)
        logger.info(f"Finished batch with {len(files)} files")
        return results
    
    def _get_language_for_extension(self, extension: str) -> Optional[str]:
        """Get language for file extension."""
        # Map extensions to language keys
        ext_map = {
            '.sql': 'SQL',
        }
        return ext_map.get(extension.lower())
    
    def _save_parsed_data(self, results: List[Dict], output_dir: Path) -> None:
        """Save parsed data to storage."""
        # TODO: Define new structure for parsed results
        # Single "dependency_graph" table
        dependency_graph_data = []

        for result in results:
            file_path = result['file_path']
            language = result['language']
            
            for obj in result['objects_defined']:
                source_schema = obj['schema'] or 'DEFAULT'
                source_name = obj['name']
                source_full_name = f"{source_schema}.{source_name}"
                
                if obj['dependencies']:
                    # Add one record per dependency relationship
                    for dep in obj['dependencies']:
                        target_schema = dep['target_schema'] or 'DEFAULT'
                        target_name = dep['target_name']
                        target_full_name = f"{target_schema}.{target_name}"
                        
                        dependency_graph_data.append({
                            # Source object info
                            'source_object_id': source_full_name,
                            'source_name': source_name,
                            'source_schema': source_schema,
                            'source_type': obj['object_type'],
                            'source_file_path': file_path,
                            'source_start_line': obj['start_line'],
                            'source_end_line': obj['end_line'],
                            
                            # Target object info
                            'target_object_id': target_full_name,
                            'target_name': target_name,
                            'target_schema': target_schema,
                            'target_type': dep['target_object_type'],
                            
                            # Dependency info
                            'dependency_type': dep['dependency_type'],
                            'dependency_line': dep['line_number'],
                            'dependency_context': dep['context'],
                            
                            # Metadata
                            'language': language,
                            'file_name': Path(file_path).name
                        })
                else:
                # Object has no dependencies - create single record with null targets
                    dependency_graph_data.append({
                        'source_object_id': source_full_name,
                        'source_name': source_name,
                        'source_schema': source_schema,
                        'source_type': obj['object_type'],
                        'source_file_path': file_path,
                        'source_start_line': obj['start_line'],
                        'source_end_line': obj['end_line'],
                        'target_object_id': None,
                        'target_name': None,
                        'target_schema': None,
                        'target_type': None,
                        'dependency_type': None,
                        'dependency_line': None,
                        'dependency_context': None,
                        'language': language,
                        'file_name': Path(file_path).name
                    })

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dependency_graph_{timestamp}.parquet"
        output_path = output_dir / filename
        
        # Save parquet file
        df = pd.DataFrame(dependency_graph_data)
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        self.logger.info(f"Saved dependency graph to: {output_path}")

def main():
    """Main entry point."""
    cli = DependencyAnalysisCLI()
    cli.main()

if __name__ == '__main__':
    main()
    