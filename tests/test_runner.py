#!/usr/bin/env python3
"""
Comprehensive test runner for Reddit Mention Tracker.
Provides detailed test execution, reporting, and coverage analysis.
"""
import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pytest
import coverage


class TestRunner:
    """Advanced test runner with comprehensive reporting and analysis."""
    
    def __init__(self, project_root: str = None):
        """Initialize test runner."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test categories
        self.test_categories = {
            'unit': ['test_database.py', 'test_scraper.py', 'test_analytics.py', 'test_ui.py'],
            'integration': ['test_integration.py'],
            'performance': ['test_performance.py'],
            'security': ['test_security.py'],
            'api': ['test_api.py']
        }
        
        # Coverage configuration
        self.coverage = coverage.Coverage(
            source=[str(self.project_root)],
            omit=[
                '*/tests/*',
                '*/venv/*',
                '*/env/*',
                '*/__pycache__/*',
                '*/migrations/*',
                'setup.py'
            ]
        )
    
    def setup_environment(self) -> bool:
        """Setup test environment and dependencies."""
        print("Setting up test environment...")
        
        try:
            # Check if pytest is available
            result = subprocess.run([sys.executable, '-m', 'pytest', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("Pytest not available")
                return False
            
            print("Test environment setup complete")
            return True
            
        except Exception as e:
            print(f"Failed to setup test environment: {e}")
            return False
    
    def run_tests(self, 
                  categories: List[str] = None,
                  test_files: List[str] = None,
                  parallel: bool = True,
                  coverage_enabled: bool = True,
                  verbose: bool = True,
                  markers: str = None) -> Dict:
        """Run tests with specified configuration."""
        
        print("Starting test execution...")
        start_time = time.time()
        
        # Determine which tests to run
        if test_files:
            test_paths = [str(self.test_dir / f) for f in test_files]
        elif categories:
            test_paths = []
            for category in categories:
                if category in self.test_categories:
                    test_paths.extend([str(self.test_dir / f) for f in self.test_categories[category]])
        else:
            test_paths = [str(self.test_dir)]
        
        # Build pytest command
        cmd = [sys.executable, '-m', 'pytest']
        cmd.extend(test_paths)
        
        # Add pytest options
        if verbose:
            cmd.append('-v')
        
        if parallel and not coverage_enabled:
            cmd.extend(['-n', 'auto'])  # Parallel execution
        
        if markers:
            cmd.extend(['-m', markers])
        
        # Add reporting options
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_report = self.reports_dir / f"test_report_{timestamp}.html"
        junit_report = self.reports_dir / f"junit_report_{timestamp}.xml"
        
        cmd.extend([
            '--html', str(html_report),
            '--self-contained-html',
            '--junit-xml', str(junit_report)
        ])
        
        # Coverage options
        if coverage_enabled:
            coverage_report = self.reports_dir / f"coverage_report_{timestamp}.html"
            cmd.extend([
                '--cov', str(self.project_root),
                '--cov-report', 'html:' + str(coverage_report),
                '--cov-report', 'term-missing',
                '--cov-report', 'json:' + str(self.reports_dir / f"coverage_{timestamp}.json")
            ])
        
        # Execute tests
        try:
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            execution_time = time.time() - start_time
            
            # Parse results
            test_results = self._parse_test_results(result, execution_time, timestamp)
            
            # Generate summary report
            self._generate_summary_report(test_results, timestamp)
            
            return test_results
            
        except Exception as e:
            print(f"Test execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_test_results(self, result: subprocess.CompletedProcess, 
                           execution_time: float, timestamp: str) -> Dict:
        """Parse test execution results."""
        
        output_lines = result.stdout.split('\n')
        
        # Extract test statistics
        stats = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'warnings': 0
        }
        
        # Parse pytest output for statistics
        # Look for the final summary line like "6 passed, 130 deselected, 15 warnings in 8.34s"
        for line in output_lines:
            line = line.strip()
            
            # Look for the final summary line
            if ('passed' in line or 'failed' in line or 'error' in line) and 'in ' in line and 's' in line:
                # Parse different patterns
                parts = line.replace(',', '').split()
                
                for i, part in enumerate(parts):
                    if i > 0:  # Make sure there's a number before the keyword
                        try:
                            count = int(parts[i-1])
                            if part == 'passed':
                                stats['passed'] = count
                            elif part == 'failed':
                                stats['failed'] = count
                            elif part in ['skipped', 'deselected']:
                                stats['skipped'] += count  # Combine skipped and deselected
                            elif part in ['error', 'errors']:
                                stats['errors'] = count
                            elif part in ['warning', 'warnings']:
                                stats['warnings'] = count
                        except (ValueError, IndexError):
                            continue
        
        # Calculate total tests (excluding deselected/skipped for main count)
        stats['total_tests'] = stats['passed'] + stats['failed'] + stats['errors']
        
        # Calculate success rate
        success_rate = (stats['passed'] / stats['total_tests'] * 100) if stats['total_tests'] > 0 else 100.0
        
        # Determine overall success
        overall_success = result.returncode == 0 and stats['failed'] == 0 and stats['errors'] == 0
        
        return {
            'success': overall_success,
            'return_code': result.returncode,
            'execution_time': execution_time,
            'timestamp': timestamp,
            'statistics': stats,
            'success_rate': success_rate,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'reports': {
                'html': f"test_report_{timestamp}.html",
                'junit': f"junit_report_{timestamp}.xml",
                'coverage': f"coverage_report_{timestamp}.html"
            }
        }
    
    def _generate_summary_report(self, results: Dict, timestamp: str):
        """Generate comprehensive summary report."""
        
        summary = {
            'test_run': {
                'timestamp': timestamp,
                'execution_time': results['execution_time'],
                'success': results['success'],
                'return_code': results['return_code']
            },
            'statistics': results['statistics'],
            'success_rate': results['success_rate'],
            'reports': results['reports'],
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': str(self.project_root)
            }
        }
        
        # Save JSON summary
        summary_file = self.reports_dir / f"test_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary to console
        self._print_summary(results)
    
    def _print_summary(self, results: Dict):
        """Print test summary to console."""
        
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)
        
        stats = results['statistics']
        
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        print(f"Total Tests: {stats['total_tests']}")
        print(f"Passed: {stats['passed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Errors: {stats['errors']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        
        # Status indicator
        if results['success']:
            print("Overall Status: PASSED")
        else:
            print("Overall Status: FAILED")
        
        # Report locations
        print(f"\nReports saved to: {self.reports_dir}")
        for report_type, filename in results['reports'].items():
            print(f"   {report_type.upper()}: {filename}")
        
        print("="*80)
    
    def run_performance_tests(self) -> Dict:
        """Run performance-specific tests."""
        print("Running performance tests...")
        
        return self.run_tests(
            categories=['performance'],
            markers='performance',
            parallel=False,  # Performance tests should run sequentially
            coverage_enabled=False  # Coverage can affect performance measurements
        )
    
    def run_security_tests(self) -> Dict:
        """Run security-specific tests."""
        print("Running security tests...")
        
        return self.run_tests(
            categories=['security'],
            markers='security',
            verbose=True
        )
    
    def run_integration_tests(self) -> Dict:
        """Run integration tests."""
        print("Running integration tests...")
        
        return self.run_tests(
            categories=['integration'],
            markers='integration',
            parallel=False  # Integration tests may have dependencies
        )
    
    def run_smoke_tests(self) -> Dict:
        """Run smoke tests for quick validation."""
        print("Running smoke tests...")
        
        return self.run_tests(
            markers='smoke',
            parallel=False,  # Disable parallel execution for smoke tests
            coverage_enabled=False
        )
    
    def run_full_test_suite(self) -> Dict:
        """Run complete test suite with all categories."""
        print("Running full test suite...")
        
        # Run tests in order of importance
        results = {}
        
        # 1. Unit tests
        print("\n1. Running unit tests...")
        results['unit'] = self.run_tests(categories=['unit'])
        
        # 2. Integration tests
        print("\n2. Running integration tests...")
        results['integration'] = self.run_integration_tests()
        
        # 3. Performance tests
        print("\n3. Running performance tests...")
        results['performance'] = self.run_performance_tests()
        
        # 4. Security tests
        print("\n4. Running security tests...")
        results['security'] = self.run_security_tests()
        
        # Generate combined report
        self._generate_combined_report(results)
        
        return results
    
    def _generate_combined_report(self, results: Dict):
        """Generate combined report for full test suite."""
        
        combined_stats = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0
        }
        
        total_time = 0
        all_success = True
        
        for category, result in results.items():
            if result.get('success'):
                stats = result.get('statistics', {})
                for key in combined_stats:
                    combined_stats[key] += stats.get(key, 0)
                total_time += result.get('execution_time', 0)
            else:
                all_success = False
        
        success_rate = (combined_stats['passed'] / combined_stats['total_tests'] * 100) if combined_stats['total_tests'] > 0 else 0
        
        print("\n" + "="*80)
        print("🏆 FULL TEST SUITE SUMMARY")
        print("="*80)
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"Total Tests: {combined_stats['total_tests']}")
        print(f"Passed: {combined_stats['passed']}")
        print(f"Failed: {combined_stats['failed']}")
        print(f"Skipped: {combined_stats['skipped']}")
        print(f"Errors: {combined_stats['errors']}")
        print(f"Overall Success Rate: {success_rate:.1f}%")
        
        if all_success and success_rate >= 95:
            print("Overall Status: EXCELLENT")
        elif all_success and success_rate >= 80:
            print("Overall Status: GOOD")
        elif success_rate >= 60:
            print("Overall Status: NEEDS IMPROVEMENT")
        else:
            print("Overall Status: CRITICAL")
        
        print("="*80)
    
    def generate_coverage_report(self) -> Dict:
        """Generate detailed coverage report."""
        print("Generating coverage report...")
        
        try:
            # Start coverage
            self.coverage.start()
            
            # Run tests with coverage
            result = self.run_tests(coverage_enabled=True, parallel=False)
            
            # Stop coverage and save
            self.coverage.stop()
            self.coverage.save()
            
            # Generate reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # HTML report
            html_dir = self.reports_dir / f"coverage_html_{timestamp}"
            self.coverage.html_report(directory=str(html_dir))
            
            # XML report
            xml_file = self.reports_dir / f"coverage_{timestamp}.xml"
            self.coverage.xml_report(outfile=str(xml_file))
            
            # Get coverage percentage
            total_coverage = self.coverage.report()
            
            return {
                'success': True,
                'coverage_percentage': total_coverage,
                'html_report': str(html_dir),
                'xml_report': str(xml_file)
            }
            
        except Exception as e:
            print(f"Coverage report generation failed: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """Main entry point for test runner."""
    
    parser = argparse.ArgumentParser(description="Reddit Mention Tracker Test Runner")
    parser.add_argument('--category', choices=['unit', 'integration', 'performance', 'security', 'api'], 
                       help='Run specific test category')
    parser.add_argument('--file', help='Run specific test file')
    parser.add_argument('--markers', help='Run tests with specific markers')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel execution')
    parser.add_argument('--no-coverage', action='store_true', help='Disable coverage reporting')
    parser.add_argument('--smoke', action='store_true', help='Run smoke tests only')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--setup-only', action='store_true', help='Setup environment only')
    parser.add_argument('--project-root', help='Project root directory')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(args.project_root)
    
    # Setup environment
    if not runner.setup_environment():
        sys.exit(1)
    
    if args.setup_only:
        print("✅ Environment setup complete")
        return
    
    # Run tests based on arguments
    try:
        if args.full:
            results = runner.run_full_test_suite()
        elif args.smoke:
            results = runner.run_smoke_tests()
        elif args.category == 'performance':
            results = runner.run_performance_tests()
        elif args.category == 'security':
            results = runner.run_security_tests()
        elif args.category == 'integration':
            results = runner.run_integration_tests()
        elif args.file:
            results = runner.run_tests(test_files=[args.file])
        elif args.category:
            results = runner.run_tests(categories=[args.category])
        else:
            # Default: run unit tests
            results = runner.run_tests(categories=['unit'])
        
        # Exit with appropriate code
        if isinstance(results, dict) and results.get('success'):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 